"""Static MLC input parsing; geometry integration is a separate step."""
import csv
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, RTPlanStorage, generate_uid
from minibeam.geometry.mlc_shape import read_csv_shape, read_rtplan_shape


def test_example_csv():
    shape = read_csv_shape('minibeam/geometry/mlc_shape_example.csv')
    assert len(shape.pairs) == 80
    assert shape.pairs[0].lower_mm == -200
    assert shape.pairs[-1].upper_mm == 200
    assert shape.pairs[39].bank_x1_mm == -26.25
    assert shape.pairs[29].bank_x2_mm == -150


def test_csv_rejects_inconsistent_width(tmp_path):
    path = tmp_path/'leaf.csv'
    path.write_text('leaf_pair,offset_mm,field_size_cm,bank_x1_cm,bank_x2_cm\n1,-2.5,1,-1,1\n2,2.5,0,0,0\n')
    with pytest.raises(ValueError, match='field size'):
        read_csv_shape(path)


def test_rtplan_inherits_mlc_positions(tmp_path):
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = RTPlanStorage; meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(tmp_path/'plan.dcm'), {}, file_meta=meta, preamble=b'\0'*128)
    beam = Dataset(); beam.BeamNumber = 3
    device = Dataset(); device.RTBeamLimitingDeviceType = 'MLCX'; device.NumberOfLeafJawPairs = 2
    device.LeafPositionBoundaries = [-5, 0, 5]; beam.BeamLimitingDeviceSequence = [device]
    first = Dataset(); second = Dataset(); pos = Dataset(); pos.RTBeamLimitingDeviceType = 'MLCX'
    pos.LeafJawPositions = [-4, -2, 4, 2]; first.BeamLimitingDevicePositionSequence = [pos]
    beam.ControlPointSequence = [first, second]; ds.BeamSequence = [beam]
    ds.save_as(ds.filename, enforce_file_format=True)
    shape = read_rtplan_shape(ds.filename, beam_number=3, control_point_index=1)
    assert [(p.bank_x1_mm, p.bank_x2_mm) for p in shape.pairs] == [(-4, 4), (-2, 2)]
    with pytest.raises(ValueError, match='out of range'):
        read_rtplan_shape(ds.filename, beam_number=3, control_point_index=2)


def test_resolved_leaves_and_touching():
    import numpy as np
    from minibeam.geometry.configuration import GeometryConfig
    from minibeam.geometry.assembly import resolve
    c=GeometryConfig.load().data
    c['mlc']['model']='leaves'
    g,_,envelopes,text=resolve(GeometryConfig.from_data(c),1000.)
    leaves=g.leaf_shape['leaves']
    assert len(leaves)==160 and text.count('Type = "G4ExtrudedSolid"')>=160
    assert g.mlc is None
    assert all(p['y_half']>0 for p in leaves)
    assert leaves[0]['adjustment_mm']<=0 and leaves[1]['adjustment_mm']>=0
    assert abs(leaves[0]['adjustment_mm']+leaves[1]['adjustment_mm'])<1e-12
    assert len(g.leaf_shape['sha256'])==64


def test_leaf_reports_and_solids(case,tmp_path):
    from minibeam.geometry.configuration import GeometryConfig
    from minibeam.geometry.assembly import TreatmentHead
    from minibeam.geometry.diagnostics import material_solids
    from minibeam.geometry.report import write_geometry_report
    from minibeam.geometry.patient_report import write_patient_geometry_report
    ct,plan,cst,stf,engine=case
    data=GeometryConfig.load().data; data['mlc']['model']='leaves'
    head=TreatmentHead(GeometryConfig.from_data(data))
    solids=material_solids(head,stf.beams[0])
    assert len([s for s in solids if s[0].startswith('MLCLeaf')])==160
    write_geometry_report(ct,stf,head,tmp_path/'water.pdf')
    write_patient_geometry_report(ct,cst,stf,head,tmp_path/'patient.pdf',target_names=('TARGET',))
    assert (tmp_path/'water.pdf').stat().st_size>1000
    assert (tmp_path/'patient.pdf').stat().st_size>1000


@pytest.mark.parametrize('model', ['simple', 'banks'])
def test_simple_model_and_legacy_alias(model):
    from minibeam.geometry.configuration import GeometryConfig
    from minibeam.geometry.models import BeamGeometry
    from minibeam.geometry.assembly import resolve
    data=GeometryConfig.load().data
    assert data['mlc']['model']=='simple'
    reference=resolve(GeometryConfig.from_data(data),1000.)[3]
    data['mlc']['model']=model
    geometry=BeamGeometry.from_config(GeometryConfig.from_data(data))
    assert resolve(geometry.as_config(),1000.)[3]==reference
    # Missing model in an old configuration still uses the simple geometry.
    del data['mlc']['model']
    assert resolve(GeometryConfig.from_data(data),1000.)[3]==reference


def test_shape_path_relative_to_toml(tmp_path,monkeypatch):
    from pathlib import Path
    from minibeam.geometry.configuration import GeometryConfig
    from minibeam.geometry.assembly import resolve
    folder=tmp_path/'configuration'; folder.mkdir()
    example=Path('minibeam/geometry/mlc_shape_example.csv').read_bytes()
    (folder/'custom.csv').write_bytes(example)
    data=GeometryConfig.load().data
    data['mlc'].update(model='leaves',shape_file='custom.csv')
    (folder/'geometry.toml').write_text(GeometryConfig.from_data(data).text)
    monkeypatch.chdir(tmp_path)
    config=GeometryConfig.load(folder/'geometry.toml')
    assert config.data['mlc']['shape_file']==str(folder/'custom.csv')
    assert len(resolve(config,1000.)[0].leaf_shape['leaves'])==160
