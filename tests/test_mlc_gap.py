"""Physical tip separation, rigid translation and saved geometry regressions."""
import numpy as np
import pytest
from minibeam.geometry.configuration import GeometryConfig
from minibeam.geometry.models import BeamGeometry
from minibeam.geometry import mlc_shape


def config():
    data=GeometryConfig.load().data
    data['mlc']['model']='leaves'
    return data


@pytest.mark.parametrize('rounded',[True,False])
@pytest.mark.parametrize('edges',[(0,0),(20,20),(-20,-20),(-.25,.25),(-50,50),(-150,-150)])
def test_minimum_and_rigid_translation(monkeypatch,rounded,edges):
    pair=mlc_shape.LeafPair(1,-2.5,2.5,*edges)
    monkeypatch.setattr(mlc_shape,'read_csv_shape',lambda _:mlc_shape.MLCShape((pair,),'csv','synthetic'))
    c=config()['mlc'];c.update(round_tip_enabled=rounded, tip_center_z_offset_mm=-7.37)
    c['minimum_leaf_gap_mm']=0.
    before=mlc_shape.resolve_leaves(c,1000.)
    c['minimum_leaf_gap_mm']=1.
    after=mlc_shape.resolve_leaves(c,1000.)
    gap=after['pair_gaps'][0]
    assert gap['resolved_gap_mm']==pytest.approx(max(1.,gap['original_gap_mm']))
    assert gap['negative_leaf_translation_mm']==-gap['positive_leaf_translation_mm']
    expected=max(0.,1-gap['original_gap_mm'])/2
    assert after['largest_leaf_adjustment_mm']==pytest.approx(expected)
    # Every polygon vertex moves rigidly; length, tip shape and Z stay unchanged.
    for old,new,sign in zip(before['leaves'],after['leaves'],[-1,1]):
        np.testing.assert_allclose(np.asarray(new['polygon']),np.asarray(old['polygon'])+[sign*expected,0],atol=1e-12)
        assert new['requested_position_mm']==old['requested_position_mm']
    # Independently find material-facing polygon boundaries at every axial level.
    polygons=[np.array(l['polygon']) for l in after['leaves']]
    gaps=[]
    for z in np.unique(polygons[0][:,1]):
        negative=polygons[0][polygons[0][:,1]==z,0].max()
        positive=polygons[1][polygons[1][:,1]==z,0].min()
        gaps.append(positive-negative)
    assert min(gaps)==pytest.approx(gap['resolved_gap_mm'],abs=1e-10)
    assert min(gaps)>=1-1e-10


@pytest.mark.parametrize('bad',[True,False,-1.,float('nan'),float('inf'),'1'])
def test_invalid_gap(bad):
    data=config();data['mlc']['minimum_leaf_gap_mm']=bad
    with pytest.raises(ValueError):
        BeamGeometry.model_validate(data)
    with pytest.raises(ValueError):
        GeometryConfig.from_data(data).validate()
    with pytest.raises(ValueError,match='minimum_leaf_gap'):
        mlc_shape.resolve_leaves(data['mlc'],1000.)


@pytest.mark.parametrize('token',['true','false','-1','nan','inf'])
def test_invalid_gap_toml(tmp_path,token):
    path=tmp_path/'invalid.toml'
    path.write_text(GeometryConfig.load().text.replace('minimum_leaf_gap_mm = 1.0',
                                                      f'minimum_leaf_gap_mm = {token}'))
    with pytest.raises(ValueError,match='minimum_leaf_gap_mm'):
        GeometryConfig.load(path)


def test_legacy_zero_and_simple_unchanged():
    from minibeam.geometry.assembly import resolve
    data=config();data['mlc']['minimum_leaf_gap_mm']=0.
    expected=resolve(GeometryConfig.from_data(data),1000.)[3]
    del data['mlc']['minimum_leaf_gap_mm']
    assert BeamGeometry.model_validate(data).mlc.minimum_leaf_gap_mm==0
    assert resolve(GeometryConfig.from_data(data),1000.)[3]==expected
    data['mlc']['model']='simple'
    expected=resolve(GeometryConfig.from_data(data),1000.)[3]
    data['mlc']['minimum_leaf_gap_mm']=1.
    assert resolve(GeometryConfig.from_data(data),1000.)[3]==expected


def test_preparation_records_and_fingerprint(case,tmp_path,capsys):
    import json
    from minibeam import TOPASPhotonEngine
    ct,plan,cst,stf,_=case
    data=config();data['aperture']['enabled']=False
    records=[]
    for gap in (0.,1.):
        data['mlc']['minimum_leaf_gap_mm']=gap
        path=tmp_path/f'geometry{gap}.toml';path.write_text(GeometryConfig.from_data(data).text)
        engine=TOPASPhotonEngine(plan,water=True,geometry_config=str(path))
        root=engine.prepare_jobs(ct,cst,stf,bundle_dir=tmp_path/f'run{gap}')
        manifest=json.loads((root/'manifest.json').read_text())
        axial=manifest['jobs'][0]['devices'][0]['parameters']['mlc_jaws']['mlc_axial_reference']
        assert axial['body_center_from_source_mm']==pytest.approx(356.8)
        assert axial['tip_curvature_center_from_source_mm']==pytest.approx(349.3)
        record=manifest['jobs'][0]['devices'][0]['parameters']['mlc_jaws']['leaf_shape']
        assert record['minimum_leaf_gap_mm']==gap
        assert min(p['resolved_gap_mm'] for p in record['pair_gaps'])>=gap-1e-10
        assert all('adjustment_mm' in l for l in record['leaves'])
        records.append(manifest['request_id'])
    assert records[0]!=records[1]
    out=capsys.readouterr().out
    assert 'MLC minimum physical tip gap: 1 mm at bank' in out
    assert 'Adjusted pairs:' in out and 'largest per-leaf shift:' in out


@pytest.mark.parametrize('patient',[False,True])
def test_gap_report(case,tmp_path,monkeypatch,patient):
    from matplotlib.axes import Axes
    from minibeam.geometry.assembly import TreatmentHead
    from minibeam.geometry.report import write_geometry_report
    from minibeam.geometry.patient_report import write_patient_geometry_report
    text=[];original=Axes.text
    def capture(self,x,y,s,*args,**kwargs):
        text.append(str(s))
        return original(self,x,y,s,*args,**kwargs)
    monkeypatch.setattr(Axes,'text',capture)
    ct,_,cst,stf,_=case
    head=TreatmentHead(GeometryConfig.from_data(config()))
    if patient:
        write_patient_geometry_report(ct,cst,stf,head,tmp_path/'patient.pdf',target_names=('TARGET',))
    else:
        write_geometry_report(ct,stf,head,tmp_path/'water.pdf')
    assert any('MLC minimum physical tip gap: 1 mm at bank' in s for s in text)
    assert any('Adjusted pairs:' in s for s in text)
    assert any('MLC body center: 356.8 mm from source; entrance/exit: 311.8/401.8 mm' in s for s in text)
    assert any('MLC tip-curvature center: 349.3 mm from source' in s for s in text)


@pytest.mark.topas
@pytest.mark.parametrize('model',['simple','leaves'])
def test_leaf_gap_transport(case,tmp_path,model):
    import os
    from minibeam import TOPASPhotonEngine
    from test_topas import execute_jobs
    executable=os.environ.get('PHOTON_TPS_TOPAS')
    if not executable:
        pytest.skip('Set PHOTON_TPS_TOPAS for local overlap/transport check')
    ct,plan,cst,stf,_=case
    shape=tmp_path/'shape.csv'
    shape.write_text('leaf_pair,offset_mm,field_size_cm,bank_x1_cm,bank_x2_cm\n'
                     '1,-7.5,0,0,0\n2,-2.5,0.05,-0.025,0.025\n'
                     '3,2.5,1,-0.5,0.5\n4,7.5,0,2,2\n')
    data=config();data['mlc'].update(shape_file=str(shape),model=model)
    data['aperture']['enabled']=False
    path=tmp_path/'geometry.toml';path.write_text(GeometryConfig.from_data(data).text)
    engine=TOPASPhotonEngine(plan,water=True,geometry_config=str(path),histories=100)
    root=engine.prepare_jobs(ct,cst,stf)
    outputs=execute_jobs(root,executable)
    assert all('Overlap is detected' not in s for s in outputs)
    assert engine.collect_results(root).physical_dose.flat[0].shape[1]==2
