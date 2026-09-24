"""Resolve configured hardware once per beam, independent of steering rays."""
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace
import math
import numpy as np
from .mlc_jaws import (JawStage, RoundedMLCTip, _rounded_mlc_polygon,
                       _render_rounded_mlc_solid, _render_jaw_solid)
from .apertures import APERTURE_TYPES
from .spatial import beam_basis, corners, overlaps_box
from ..topas.contracts import ParameterFragment


class GeometryCollisionError(ValueError):
    """Expected placement conflict, optionally accompanied by saved diagnostics."""


def serializable(value):
    if is_dataclass(value):return serializable(asdict(value))
    if isinstance(value,SimpleNamespace):return serializable(vars(value))
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serializable(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    return value


def resolve(config,sad):
    data=config.data; field=data['field']
    if not math.isfinite(sad) or sad<=0:raise ValueError('Beam SAD must be positive')
    g=SimpleNamespace(source_z=-sad,field_aperture_outer_width=field['outer_width_mm'],
        field_aperture_outer_height=field['outer_height_mm'],field_aperture_material=field['material'],
        mlc=None,lower_jaw=None,mlc_round_tip=None,mlc_round_tip_enabled=data['mlc']['round_tip_enabled'])
    envelopes=[]; texts=[]; stages=[]
    if not data['assembly']['enabled']: return g,None,[],''
    for key,axis,opening in [('mlc','X',field['width_mm']),('jaws','Y',field['height_mm'])]:
        c=data[key]
        if not c['enabled']:continue
        d,t=c['source_to_center_mm'],c['thickness_mm']
        if d-t/2<=0 or d+t/2>=sad:raise ValueError(f'{key} must lie between source and isocenter')
        z=d-sad
        stage=JawStage(axis,d,z,t,z-t/2,z+t/2,opening*(d-t/2)/sad,opening*(d+t/2)/sad)
        if key=='mlc':
            g.mlc=stage
            if g.mlc_round_tip_enabled:
                radius=c['tip_radius_mm'];zc=z+c['tip_center_z_offset_mm']
                if radius<=max(abs(stage.entrance_z-zc),abs(stage.exit_z-zc)):
                    raise ValueError('MLC tip radius does not span thickness')
                slope=opening/2/sad
                xc=slope*(zc+sad)+radius*math.sqrt(1+slope*slope)
                tangent_distance=((zc+sad)+slope*xc)/(1+slope*slope)
                tangent_z=tangent_distance-sad
                if not stage.entrance_z-1e-9<=tangent_z<=stage.exit_z+1e-9:
                    raise ValueError('MLC tangent lies outside bank thickness')
                g.mlc_round_tip=RoundedMLCTip(radius,xc,zc,c['tip_center_z_offset_mm'],tangent_z,slope*tangent_distance)
                def inner(at):return xc-math.sqrt(max(0,radius**2-(at-zc)**2))
                g.mlc=JawStage(axis,d,z,t,z-t/2,z+t/2,2*inner(z-t/2),2*inner(z+t/2))
                # The largest inner face can occur at either entrance or exit.
                if min(inner(z-t/2),inner(z+t/2),inner(min(max(zc,z-t/2),z+t/2)))<=0:
                    raise ValueError('MLC banks overlap across the central opening')
            stage=g.mlc
        else:g.lower_jaw=stage
        outer=field['outer_width_mm'] if axis=='X' else field['outer_height_mm']
        if max(stage.entrance_opening,stage.exit_opening)+2*field['side_margin_mm']>=outer:
            raise ValueError(f'{key} opening exceeds outer dimensions/side margin')
        stages.append((key,stage.entrance_z,stage.exit_z))
        envelopes.append({'name':key,'center':[0,0,z],
            'half_size':[field['outer_width_mm']/2,field['outer_height_mm']/2,t/2], 'rotation':np.eye(3).tolist()})
        for positive in (False,True):
            if key=='mlc' and g.mlc_round_tip_enabled:text=_render_rounded_mlc_solid(g,positive)
            else:text=_render_jaw_solid(g,stage,positive,'MLC' if key=='mlc' else 'LowerJaw',key)
            texts.append(text.replace('Parent = "World"','Parent = "TreatmentHead"'))
    aperture=None; aperture_type=None
    if data['aperture']['enabled']:
        aperture_type=APERTURE_TYPES[data['aperture']['type']]
        aperture=aperture_type.resolve(data['aperture'],sad)
        envelope=aperture_type.envelope(aperture)
        points=corners(envelope)
        lo,hi=points[:,2].min(),points[:,2].max()
        if lo<=-sad or hi>=0:raise ValueError('Rotated aperture must lie between source and isocenter')
        stages.append(('aperture',lo,hi));envelopes.append(envelope)
        texts.append(aperture_type.render(aperture,'TreatmentHead'))
    for previous,current in zip(stages,stages[1:]):
        if previous[2]>current[1]+1e-7:
            raise ValueError(f'Device stages overlap or are out of order: {previous[0]}, {current[0]}')
    return g,aperture,envelopes,'\n'.join(texts)


class TreatmentHead:
    name='TreatmentHead'
    def __init__(self,config):
        self.config=config
        self._resolved={}

    def config_for(self, beam):
        return beam.geometry.as_config() if hasattr(beam, 'geometry') else self.config

    def resolved(self,beam):
        sad=float(beam.sad)
        if not np.isclose(np.linalg.norm(beam.source_point),sad,atol=1e-6):
            raise ValueError('Configured focused devices require native source-to-isocenter distance equal to SAD')
        config = self.config_for(beam)
        key = (config.sha256, sad)
        if key not in self._resolved:self._resolved[key]=resolve(config,sad)
        return self._resolved[key]

    def render(self,context):
        config = self.config_for(context.beam)
        g,aperture,envelopes,text=self.resolved(context.beam)
        basis=beam_basis(context.beam)
        origin=np.asarray(context.beam.iso_center)-context.patient_center
        points=np.concatenate([corners(e) for e in envelopes]) if envelopes else np.zeros((1,3))
        points=points@basis.T+origin
        prefix='\n'.join(['b:Ge/CheckForOverlaps = "True"',
                          'b:Ge/QuitIfOverlapDetected = "True"',
                          'i:Ge/CheckForOverlapsResolution = 10000',
                          's:Ge/TreatmentHead/Type = "Group"', 's:Ge/TreatmentHead/Parent = "BeamFrame"',
                          *[f'd:Ge/TreatmentHead/Trans{a} = 0 mm' for a in 'XYZ']])
        return ParameterFragment(prefix+'\n'+text,
            record={'configuration_sha256':config.sha256,'field':config.data['field'],
                    'mlc_jaws':serializable(g),'aperture_type':config.data['aperture']['type'] if aperture else None,
                    'aperture':serializable(aperture),'envelopes':envelopes},
            bounds_topas_mm=(points.min(axis=0),points.max(axis=0)))

    def validate_patient(self,ct,beam,beam_index):
        from .coordinates import patient_center
        _,_,envelopes,_=self.resolved(beam)
        basis=beam_basis(beam); origin=np.asarray(beam.iso_center)-patient_center(ct)
        half=np.asarray(ct.size)*ct.grid.resolution_vector/2
        for e in envelopes:
            if overlaps_box(basis@e['center']+origin,basis@np.asarray(e['rotation']),e['half_size'],half):
                raise GeometryCollisionError(f'Beam {beam_index}: conservative {e["name"]} envelope overlaps CT transport box; adjust geometry/placement explicitly')
