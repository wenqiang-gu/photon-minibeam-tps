"""Entrance dimensions are display annotations, independent of source replay."""
import numpy as np
import pytest
from pyRadPlan import generate_stf
from minibeam.geometry.assembly import TreatmentHead
from minibeam.geometry.models import BeamGeometry
from minibeam.geometry.report import entrance_distances, draw_entrance_dimensions, write_geometry_report
from minibeam.geometry.patient_report import write_patient_geometry_report


@pytest.mark.parametrize('model',['simple','leaves'])
@pytest.mark.parametrize('thickness,expected',[(77.,432.),(40.,450.5)])
def test_entrance_reference(case,model,thickness,expected):
    geometry=BeamGeometry.from_config().replace(mlc={'model':model},jaws={'thickness_mm':thickness})
    head=TreatmentHead(geometry.as_config())
    assert dict(entrance_distances(head,case[3].beams[0]))==pytest.approx({'MLC':311.8,'jaw':expected})
    disabled=geometry.replace(mlc={'enabled':False})
    assert entrance_distances(TreatmentHead(disabled.as_config()),case[3].beams[0])==[('jaw',expected)]
    assert entrance_distances(TreatmentHead(geometry.replace(assembly={'enabled':False}).as_config()),case[3].beams[0])==[]


def test_dimension_endpoints_preserve_foreshortening():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    source=np.array([10.,20.]);direction=np.array([.6,0.])
    draw_entrance_dimensions(ax,source,direction,[('MLC',311.8),('jaw',450.)],np.array([[0,0],[400,100]]))
    arrows=[a for a in ax.texts if getattr(a,'arrow_patch',None) is not None]
    for arrow,distance in zip(arrows,[311.8,450.]):
        np.testing.assert_allclose(np.asarray(arrow.xy)-arrow.xyann,direction*distance)
    assert any('311.8 mm' in a.get_text() for a in ax.texts)
    plt.close(fig)


@pytest.mark.parametrize('patient',[False,True])
@pytest.mark.parametrize('model',['simple','leaves'])
def test_dimension_reports(case,tmp_path,monkeypatch,patient,model):
    from matplotlib.backends.backend_pdf import PdfPages
    import minibeam.geometry.patient_report as patient_module
    import minibeam.geometry.report as water_module
    ct,plan,cst,_,_=case
    plan.prop_stf.update(gantry_angles=[0.,45.,90.,135.],couch_angles=[0.,0.,0.,25.])
    stf=generate_stf(ct,cst,plan)
    head=TreatmentHead(BeamGeometry.from_config().replace(mlc={'model':model}).as_config())
    calls=[]
    def capture(ax,source,direction,distances,occupied):
        calls.append((np.asarray(source),np.asarray(direction),distances))
        return draw_entrance_dimensions(ax,source,direction,distances,occupied)
    monkeypatch.setattr(patient_module if patient else water_module,'draw_entrance_dimensions',capture)
    save=PdfPages.savefig
    def check_and_save(self,fig,*args,**kwargs):
        fig.canvas.draw()
        for ax in fig.axes:
            for text in ax.texts:
                if text.get_text().startswith('Source →'):
                    bounds=text.get_window_extent(fig.canvas.get_renderer())
                    assert bounds.x0>=0 and bounds.y0>=0
                    assert bounds.x1<=fig.bbox.x1 and bounds.y1<=fig.bbox.y1
                    assert bounds.x0>=ax.bbox.x0 and bounds.y0>=ax.bbox.y0
                    assert bounds.x1<=ax.bbox.x1 and bounds.y1<=ax.bbox.y1
        return save(self,fig,*args,**kwargs)
    monkeypatch.setattr(PdfPages,'savefig',check_and_save)
    before=head.resolved(stf.beams[0])[3]
    if patient:
        # Replay plane must not become the origin of the dimension arrows.
        records=[dict(beam_index=i,model='phase_space_square_beamlet',
                      phase_space_plane_corners_lps_mm=[[0,-700,0],[10,-700,0],[10,-700,10],[0,-700,10]],
                      phase_space_plane_lps_mm=[0,-700,0]) for i in range(1,5)]
        write_patient_geometry_report(ct,cst,stf,head,tmp_path/'patient.pdf',source_records=records)
        assert len(calls)==4
        for beam,(source,direction,distances) in zip(stf.beams,calls):
            unit=-np.asarray(beam.source_point)/beam.sad
            views=[(0,1),(0,2),(1,2)]
            axes=views[int(np.argmax([np.linalg.norm(unit[list(a)]) for a in views]))]
            np.testing.assert_allclose(source,(beam.iso_center+beam.source_point)[list(axes)])
            np.testing.assert_allclose(direction,unit[list(axes)])
    else:
        write_geometry_report(ct,stf,head,tmp_path/'water.pdf')
        assert len(calls)==8
        for source,direction,_ in calls:
            np.testing.assert_allclose(source,[-1000,0])
            np.testing.assert_allclose(direction,[1,0])
    assert head.resolved(stf.beams[0])[3]==before
