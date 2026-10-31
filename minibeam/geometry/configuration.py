"""Read an immutable TOML snapshot; physical placement uses each native beam SAD."""
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import hashlib
import re
import tomllib
import math
import json

def normalize_body_dimensions(data):
    """Normalize legacy shared envelopes without mutating saved snapshots."""
    from copy import deepcopy
    data = deepcopy(data)
    field = data.get('field', {})
    keys = ('outer_width_mm', 'outer_height_mm')
    if not isinstance(field, dict):
        raise ValueError('Geometry field must be a table')
    if set(keys) & field.keys():
        if any(not isinstance(data.get(k), dict) for k in ('mlc', 'jaws')):
            raise ValueError('Legacy dimensions require mlc and jaws tables')
        if not set(keys) <= field.keys() or any(set(keys) & data[k].keys() for k in ('mlc', 'jaws')):
            raise ValueError('Mixed or incomplete body dimensions: remove shared field.outer_* and specify both dimensions in mlc and jaws, or use only the legacy shared pair')
        for section in ('mlc', 'jaws'):
            data[section].update({key: field[key] for key in keys})
        for key in keys:
            del field[key]
    return data


@dataclass(frozen=True)
class GeometryConfig:
    text: str

    @classmethod
    def load(cls, path=None):
        text = (Path(path).read_text() if path is not None else
                resources.files('minibeam.geometry').joinpath('config.toml').read_text())
        config = cls(text)
        # Report invalid numeric settings before serializing resolved asset paths.
        config.validate()
        data = config.data
        if data["mlc"]["shape_file"]:
            source = Path(data["mlc"]["shape_file"]).expanduser()
            if not source.is_absolute():
                source = (Path(path).resolve().parent if path else Path(__file__).parent) / source
            data["mlc"]["shape_file"] = str(source.resolve())
            config = cls.from_data(data)
        return config

    @classmethod
    def from_data(cls, data):
        lines = []
        for section, values in data.items():
            lines.append(f'[{section}]')
            for key, value in values.items():
                lines.append(f'{key} = {json.dumps(value, allow_nan=False)}')
        return cls('\n'.join(lines) + '\n')

    @property
    def data(self):
        data = tomllib.loads(self.text)
        if isinstance(data.get('mlc'), dict):
            for key,value in dict(model='simple',shape_source='csv',shape_file='mlc_shape_example.csv',shape_beam_number=1,shape_control_point_index=0,leaf_length_mm=155.,minimum_leaf_gap_mm=0.).items():
                data['mlc'].setdefault(key,value)
        if isinstance(data.get('aperture'), dict):
            data['aperture'].setdefault('lateral_shift_mm', 0.0)
            data['aperture'].setdefault('rotation_z_deg', 0.0)
        if isinstance(data.get('jaws'), dict):
            data['jaws'].setdefault('round_tip_enabled', False)
            data['jaws'].setdefault('tip_radius_mm', 135.0)
            data['jaws'].setdefault('tip_center_z_offset_mm', -3.5)
        return normalize_body_dimensions(data)

    @property
    def sha256(self):
        return hashlib.sha256(self.text.encode()).hexdigest()

    def validate(self):
        from .apertures import APERTURE_TYPES
        data = self.data
        allowed = {
            'assembly': {'enabled'},
            'field': {'width_mm','height_mm','material','side_margin_mm'},
            'mlc': {'enabled','source_to_center_mm','thickness_mm','outer_width_mm','outer_height_mm','round_tip_enabled','tip_radius_mm','tip_center_z_offset_mm'},
            'jaws': {'enabled','source_to_center_mm','thickness_mm','outer_width_mm','outer_height_mm','round_tip_enabled','tip_radius_mm','tip_center_z_offset_mm'},
            'aperture': set(),
        }
        if set(data) != set(allowed):
            raise ValueError('Geometry TOML requires assembly, field, mlc, jaws and aperture sections only')
        if not all(isinstance(table,dict) for table in data.values()):
            raise ValueError('Geometry configuration sections must be TOML tables')
        if not isinstance(data['aperture'].get('type'),str) or data['aperture'].get('type') not in APERTURE_TYPES:
            raise ValueError('Unknown aperture.type; currently supported: ' + ', '.join(APERTURE_TYPES))
        aperture_type=APERTURE_TYPES[data['aperture']['type']]
        allowed['mlc'] |= {'model','shape_source','shape_file','shape_beam_number','shape_control_point_index','leaf_length_mm','minimum_leaf_gap_mm'}
        allowed['aperture']=aperture_type.config_keys
        for section, keys in allowed.items():
            table=data[section]
            if not isinstance(table,dict) or set(table) != keys:
                raise ValueError(f'Invalid or missing keys in geometry [{section}]')
            for key,value in table.items():
                if key in {'model','shape_source','shape_file'}:
                    if not isinstance(value,str): raise ValueError(f'{key} must be a string')
                    if key=='shape_file' and not value.strip(): raise ValueError('shape_file must be a nonempty file path')
                    if key=='model' and value not in {'simple','banks','leaves'}: raise ValueError('Unknown MLC model')
                    if key=='shape_source' and value not in {'csv','rtplan'}: raise ValueError('Unknown MLC shape source')
                elif key in {'shape_beam_number','shape_control_point_index'}:
                    if type(value) is not int or value<0: raise ValueError(f'{key} must be a nonnegative integer')
                elif key in {'enabled','round_tip_enabled'}:
                    if type(value) is not bool: raise ValueError(f'{section}.{key} must be boolean')
                elif key == 'material':
                    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',value):
                        raise ValueError(f'{section}.material must be a TOPAS material name')
                elif key == 'type': pass
                else:
                    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                        raise ValueError(f'{section}.{key} must be finite numeric')
                    signed = key.startswith('rotation_') or key in {'tip_center_z_offset_mm', 'lateral_shift_mm'}
                    if not signed and (value < 0 or (value == 0 and key not in {'side_margin_mm','minimum_leaf_gap_mm'})):
                        raise ValueError(f'{section}.{key} must be positive')
        aperture_type.validate_config(data['aperture'])
