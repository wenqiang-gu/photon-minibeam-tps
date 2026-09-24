"""Read an immutable TOML snapshot; physical placement uses each native beam SAD."""
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import hashlib
import re
import tomllib
import math
import json

@dataclass(frozen=True)
class GeometryConfig:
    text: str

    @classmethod
    def load(cls, path=None):
        text = (Path(path).read_text() if path is not None else
                resources.files('minibeam.geometry').joinpath('config.toml').read_text())
        config = cls(text)
        config.validate()
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
        if isinstance(data.get('aperture'), dict):
            data['aperture'].setdefault('lateral_shift_mm', 0.0)
        return data

    @property
    def sha256(self):
        return hashlib.sha256(self.text.encode()).hexdigest()

    def validate(self):
        from .apertures import APERTURE_TYPES
        data = self.data
        allowed = {
            'assembly': {'enabled'},
            'field': {'width_mm','height_mm','material','outer_width_mm','outer_height_mm','side_margin_mm'},
            'mlc': {'enabled','source_to_center_mm','thickness_mm','round_tip_enabled','tip_radius_mm','tip_center_z_offset_mm'},
            'jaws': {'enabled','source_to_center_mm','thickness_mm'},
            'aperture': set(),
        }
        if set(data) != set(allowed):
            raise ValueError('Geometry TOML requires assembly, field, mlc, jaws and aperture sections only')
        if not all(isinstance(table,dict) for table in data.values()):
            raise ValueError('Geometry configuration sections must be TOML tables')
        if not isinstance(data['aperture'].get('type'),str) or data['aperture'].get('type') not in APERTURE_TYPES:
            raise ValueError('Unknown aperture.type; currently supported: ' + ', '.join(APERTURE_TYPES))
        aperture_type=APERTURE_TYPES[data['aperture']['type']]
        allowed['aperture']=aperture_type.config_keys
        for section, keys in allowed.items():
            table=data[section]
            if not isinstance(table,dict) or set(table) != keys:
                raise ValueError(f'Invalid or missing keys in geometry [{section}]')
            for key,value in table.items():
                if key in {'enabled','round_tip_enabled'}:
                    if type(value) is not bool: raise ValueError(f'{section}.{key} must be boolean')
                elif key == 'material':
                    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',value):
                        raise ValueError(f'{section}.material must be a TOPAS material name')
                elif key == 'type': pass
                else:
                    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                        raise ValueError(f'{section}.{key} must be finite numeric')
                    signed = key.startswith('rotation_') or key in {'tip_center_z_offset_mm', 'lateral_shift_mm'}
                    if not signed and (value < 0 or (value == 0 and key != 'side_margin_mm')):
                        raise ValueError(f'{section}.{key} must be positive')
        aperture_type.validate_config(data['aperture'])
