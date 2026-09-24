"""PDF schematics from the same resolved device shapes used for TOPAS."""
import os
from pathlib import Path
import tempfile
import numpy as np
from .mlc_jaws import _rounded_mlc_polygon
from .spatial import beam_basis, corners
from .coordinates import patient_center
from .apertures import APERTURE_TYPES


def write_geometry_report(ct, stf, head, path, diagnostics=None):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'minibeam-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.backends.backend_pdf import PdfPages
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.pdf',delete=False) as stream:
        temporary=Path(stream.name)
    try:
        with PdfPages(temporary,metadata={'Title':'Photon minibeam geometry','CreationDate':None}) as pdf:
            for index,beam in enumerate(stf.beams,1):
                fig=plt.figure(figsize=(13,10),layout='constrained')
                gs=fig.add_gridspec(3,2,height_ratios=(1,1,1.5))
                xz=fig.add_subplot(gs[0,:]); yz=fig.add_subplot(gs[1,:])
                entrance=fig.add_subplot(gs[2,0]); info=fig.add_subplot(gs[2,1]);info.axis('off')
                fig.suptitle(f'Photon minibeam geometry  |  Beam {index}\n'
                    f'Gantry {beam.gantry_angle:g} deg   Couch {beam.couch_angle:g} deg   SAD {beam.sad:g} mm',fontsize=16)
                g,ap,envelopes,_=head.resolved(beam) if head else (None,None,[],None)
                basis=beam_basis(beam)
                half=np.asarray(ct.size)*ct.grid.resolution_vector/2
                ct_corners=corners({'center':[0,0,0],'half_size':half,'rotation':np.eye(3)})
                ct_local=(ct_corners+patient_center(ct)-beam.iso_center)@basis
                for ax,axis,axis_name in [(xz,0,'X'),(yz,1,'Y')]:
                    ax.axhline(0,color='#718096',lw=.7)
                    ax.axvline(0,color='#167d9a',ls='--',lw=1)
                    ax.plot(-beam.sad,0,'*',color='#cb8500',ms=12)
                    ax.annotate('Source',(-beam.sad,0),xytext=(5,10),textcoords='offset points',fontsize=9)
                    ax.add_patch(Rectangle((ct_local[:,2].min(),ct_local[:,axis].min()),
                        np.ptp(ct_local[:,2]),np.ptp(ct_local[:,axis]),fc='#e8edf1',ec='#8896a0',ls='--',label='CT box projection'))
                    for e in envelopes:
                        points=corners(e)
                        from scipy.spatial import ConvexHull
                        projection=points[:,[2,axis]]
                        hull=ConvexHull(projection)
                        ax.add_patch(Polygon(projection[hull.vertices],fill=False,ec='#555555',lw=.8))
                        ax.text(np.mean(points[:,2]),points[:,axis].max()+8,e['name'],ha='center',fontsize=8)
                    if g:
                        field=head.config_for(beam).data['field']
                        w=field['width_mm'] if axis==0 else field['height_mm']
                        for sign in [-1,1]:ax.plot([-beam.sad,0],[0,sign*w/2],color='#db8d2b',lw=.9,ls='--')
                        stage=g.mlc if axis==0 else g.lower_jaw
                        if stage:
                            for sign in [-1,1]:
                                if axis==0 and g.mlc_round_tip_enabled:
                                    shape=np.asarray(_rounded_mlc_polygon(g,sign>0))
                                    points=np.column_stack((shape[:,1]+stage.center_z,shape[:,0]))
                                else:
                                    outer=field['outer_width_mm' if axis==0 else 'outer_height_mm']/2
                                    points=np.array([[stage.entrance_z,sign*stage.entrance_opening/2],
                                        [stage.entrance_z,sign*outer],[stage.exit_z,sign*outer],
                                        [stage.exit_z,sign*stage.exit_opening/2]])
                                ax.add_patch(Polygon(points,fc='#718299',ec='#374b60',lw=.6))
                    if ap and axis==0:
                        for polygon in APERTURE_TYPES[head.config_for(beam).data['aperture']['type']].drawing(ap):
                            points=np.asarray(polygon)
                            ax.add_patch(Polygon(points[:,[2,0]],fc='#bba25f',ec='#806b38',lw=.4))
                    ax.set_xlim(-beam.sad-45,max(50,ct_local[:,2].max()+20))
                    ax.set_ylim(min(-230,ct_local[:,axis].min()-30),max(240,ct_local[:,axis].max()+30))
                    ax.set_xlabel('Beam-local Z (mm); isocenter at 0',fontsize=9)
                    ax.set_ylabel(f'{axis_name} (mm)',fontsize=9);ax.grid(alpha=.15)
                if ap:
                    APERTURE_TYPES[head.config_for(beam).data['aperture']['type']].draw_entrance(ap,entrance)
                else:
                    entrance.axis('off');entrance.text(.1,.5,'No downstream aperture',fontsize=12)
                lines=['GEOMETRY RECORD', 'Schematic, not a patient surface or dose image.',
                       'Dashed gray: conservative CT box projection.', 'Dashed orange: configured field edges.']
                if head:
                    d=head.config_for(beam).data;f=d['field']
                    lines += [f"Field at isocenter: {f['width_mm']:g} x {f['height_mm']:g} mm",
                              f"Field material: {f['material']}"]
                    for title,stage in [('MLC',g.mlc),('Jaws',g.lower_jaw)]:
                        if stage:lines += [f'{title}: source-center {stage.source_to_center:g} mm; thickness {stage.thickness:g} mm',
                            f'  Entrance / exit opening: {stage.entrance_opening:.3f} / {stage.exit_opening:.3f} mm']
                    if g.mlc_round_tip:
                        tip=g.mlc_round_tip
                        lines += [f'MLC tip: radius {tip.radius:g} mm; offset {tip.center_z_offset:g} mm',
                                  f'Tangent (X,Z): ({tip.tangent_x:.3f}, {tip.tangent_z:.3f}) mm']
                    if ap:
                        lines += APERTURE_TYPES[head.config_for(beam).data['aperture']['type']].summary(ap)
                    lines += ['MLC is a two-bank approximation, not individual leaves.']
                else:lines += ['Configured assembly disabled.']
                info.text(0,1,'\n'.join(lines),va='top',fontsize=9,linespacing=1.5,family='DejaVu Sans')
                pdf.savefig(fig);plt.close(fig)
                if diagnostics is not None:
                    record = diagnostics[index-1]
                    fig, ax = plt.subplots(figsize=(13,10),layout='constrained')
                    ax.axis('off')
                    fig.suptitle(f'Patient collision diagnostics | Beam {index}',fontsize=16)
                    lines = ['Coordinates: DICOM LPS, mm. CT box intersections do not imply tissue collisions.',
                             'Redundant enclosing-volume and material checks are reported separately.',
                             'The slit enclosing box is itself a TOPAS volume, including its air cavity.', '']
                    for title, key in [('ENCLOSING VOLUMES','envelopes'),('MATERIAL SOLIDS','material_solids')]:
                        lines.append(title)
                        hits = [item for item in record[key] if item['intersects']]
                        for item in hits:
                            x,y,z = item['witness_lps_mm']
                            lines.append(f"{item['component']}: witness ({x:.3f}, {y:.3f}, {z:.3f}); "
                                         f"intersection inradius {item['intersection_inradius_mm']:.3f} mm")
                        if not hits: lines.append('No intersections found.')
                        lines.append('')
                    lines += ['Witness: a point strictly inside both volumes.',
                              'Inradius: radius of a ball inside the intersection, not penetration depth.',
                              'See collisions.json for component bounds and the CT cropping investigation.',
                              'Preparation remains blocked. No CT or geometry was automatically changed.']
                    ax.text(.02,.98,'\n'.join(lines),va='top',fontsize=10,linespacing=1.55)
                    pdf.savefig(fig);plt.close(fig)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
