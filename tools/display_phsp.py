#!/usr/bin/env python3
"""Inspect original or modified IAEA phase-space pairs without running TOPAS."""
import argparse
from pathlib import Path
import json
import sys
import tempfile
import shutil

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minibeam.sources.iaea import read_header, validated_parts

SPECIES = {'photon':1, 'electron':2, 'positron':3, 'all':None}


def values(a, plane, projection):
    u=a['u'].astype(float);v=a['v'].astype(float)
    w=np.sqrt(np.maximum(0,1-u*u-v*v))*np.sign(a['code'])
    x=a['x'].astype(float)*10;y=a['y'].astype(float)*10
    forward=w>0
    photon_forward=(a['code']==1)&forward
    radial=np.hypot(x[photon_forward],y[photon_forward])
    divergence=np.hypot(u[photon_forward],v[photon_forward])/w[photon_forward]
    # Intersections are geometric extrapolations without any transport/materials.
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        px=x[forward]+(projection-plane)*u[forward]/w[forward]
        py=y[forward]+(projection-plane)*v[forward]/w[forward]
    if not (np.isfinite(px).all() and np.isfinite(py).all()):
        raise ValueError('Nonfinite projected coordinates')
    return dict(energy=np.abs(a['energy'].astype(float)),x=x,y=y,
        angle_x=np.degrees(np.arctan2(u,w)),angle_y=np.degrees(np.arctan2(v,w)),
        weight=a['weight'].astype(float),photon_forward=photon_forward,radial=radial,divergence=divergence,projected_x=px,projected_y=py,forward=forward)


def edges(low,high,bins):
    if not np.isfinite(low):return np.linspace(-.5,.5,bins+1)
    if low==high:
        padding=max(abs(low)*.01,1e-6);low-=padding;high+=padding
    return np.linspace(low,high,bins+1)


def inspect_sources(bases, output_dir, *, species='photon', histories=None,
                    bins=200, projection_distance_mm=1000.):
    if species not in SPECIES:raise ValueError('Unknown species')
    if type(bins) is not int or not 10<=bins<=2000:raise ValueError('bins must be an integer from 10 to 2000')
    if not np.isfinite(projection_distance_mm):raise ValueError('Projection distance must be finite')
    bases=[Path(b).expanduser().resolve() for b in bases]
    bases=[b.with_suffix('') if b.suffix in ('.header','.phsp') else b for b in bases]
    if not bases or len(set(bases))!=len(bases):raise ValueError('Supply distinct input basenames')
    output=Path(output_dir).expanduser().resolve()
    if output.exists():raise ValueError(f'Output already exists: {output}; use a fresh directory')
    if any(output in b.parents for b in bases):raise ValueError('Output must not contain the inputs')
    infos=[read_header(b) for b in bases];plane=infos[0]['plane_distance_mm']
    if any(i['plane_distance_mm']!=plane or i['source_description']!=infos[0]['source_description'] for i in infos):
        raise ValueError('Input source planes/descriptions differ; inspect separate source sets separately')
    if projection_distance_mm<=plane:raise ValueError('Projection plane must be downstream of the recorded plane')
    available=sum(i['counts']['ORIG_HISTORIES'] for i in infos)
    if histories is None:histories=available
    if type(histories) is not int or not 1<=histories<=available:raise ValueError('histories must be a positive original-history prefix within the available count')
    keys=['energy','x','y','angle_x','angle_y','weight','projected_x','projected_y','radial','divergence']
    limits={k:[np.inf,-np.inf] for k in keys}
    count=forward_count=nonempty=0;last_history=None;total_weight=energy_sum=forward_weight=0.
    photon_forward_count=0;photon_forward_weight=0.;photon_excluded_count=0
    audits=[]
    def selected(a,ids):
        mask=ids<=histories
        if SPECIES[species] is not None:mask &= np.abs(a['code'].astype(int))==SPECIES[species]
        return a[mask],ids[mask]
    print(f'Inspecting {len(infos)} parts, {histories:,}/{available:,} original histories; species={species}',flush=True)
    current=None
    for index,a,ids in validated_parts(infos,audits):
        if current!=index:
            print(f'Pass 1: ranges/validation, part {index+1}/{len(infos)}',flush=True);current=index
        a,ids=selected(a,ids)
        if not len(a):continue
        d=values(a,plane,projection_distance_mm)
        photon_forward_count+=int(d['photon_forward'].sum())
        photon_forward_weight+=float(d['weight'][d['photon_forward']].sum())
        photon_excluded_count+=int(((np.abs(a['code'].astype(int))==1)&~d['photon_forward']).sum())
        nonempty+=int(np.count_nonzero(np.r_[ids[0]!=last_history,ids[1:]!=ids[:-1]]));last_history=int(ids[-1])
        count+=len(a);forward_count+=int(d['forward'].sum());total_weight+=float(d['weight'].sum())
        energy_sum+=float(np.sum(d['weight']*d['energy']));forward_weight+=float(d['weight'][d['forward']].sum())
        for k in keys:
            if len(d[k]):limits[k]=[min(limits[k][0],float(d[k].min())),max(limits[k][1],float(d[k].max()))]
    if count==0:raise ValueError('No particles of the selected species in the requested prefix')
    if total_weight<=0:raise ValueError('Selected particles have zero total statistical weight')
    arrays={k+'_edges':edges(*limits[k],bins) for k in keys}
    # Use equal physical spans for each map, including narrow/constant axes.
    for kx,ky in [('x','y'),('angle_x','angle_y'),('projected_x','projected_y')]:
        ex,ey=arrays[kx+'_edges'],arrays[ky+'_edges']
        span=max(ex[-1]-ex[0],ey[-1]-ey[0])
        for k,e in [(kx,ex),(ky,ey)]:
            center=(e[0]+e[-1])/2
            arrays[k+'_edges']=np.linspace(min(center-span/2,e[0]),max(center+span/2,e[-1]),bins+1)
    # Histogram the viewing window itself so distant outliers cannot coarsen bins.
    for key in ('projected_x','projected_y'):
        arrays[key+'_edges']=np.linspace(-100.,100.,bins+1)
    projection_inside_count=0;projection_inside_weight=0.
    arrays["radial_edges"]=np.linspace(0.,80.,bins+1)
    arrays["divergence_edges"]=np.linspace(0.,.5,bins+1)
    divergence_inside_count=0;divergence_inside_weight=0.
    # Fixed 0.5-keV detail bins make the 511-keV component visible without
    # changing or claiming an energy-based ancestry classification.
    arrays['line_energy_edges']=np.linspace(.49,.53,81)
    for k in ['energy','weight','line_energy']:arrays[k+'_counts']=np.zeros(len(arrays[k+'_edges'])-1)
    for name in ['position','angle','projection','radial_divergence']:arrays[name+'_weighted_counts']=np.zeros((bins,bins))
    second=[];current=None
    for index,a,ids in validated_parts(infos,second):
        if current!=index:
            print(f'Pass 2: histograms, part {index+1}/{len(infos)}',flush=True);current=index
        a,_=selected(a,ids)
        if not len(a):continue
        d=values(a,plane,projection_distance_mm);weights=d['weight']
        inside=(np.abs(d['projected_x'])<=100)&(np.abs(d['projected_y'])<=100)
        projection_inside_count+=int(inside.sum())
        projection_inside_weight+=float(weights[d['forward']][inside].sum())
        visible=(d['radial']<=80)&(d['divergence']<=.5)
        divergence_inside_count+=int(visible.sum())
        divergence_inside_weight+=float(weights[d['photon_forward']][visible].sum())
        arrays['energy_counts']+=np.histogram(d['energy'],arrays['energy_edges'],weights=weights)[0]
        arrays['line_energy_counts']+=np.histogram(d['energy'],arrays['line_energy_edges'],weights=weights)[0]
        # Weight histogram counts RECORDS, rather than weighting by weight again.
        arrays['weight_counts']+=np.histogram(weights,arrays['weight_edges'])[0]
        for name,kx,ky,w in [('position','x','y',weights),('angle','angle_x','angle_y',weights),
                            ('projection','projected_x','projected_y',weights[d['forward']]),
                            ('radial_divergence','radial','divergence',weights[d['photon_forward']])]:
            arrays[name+'_weighted_counts']+=np.histogram2d(d[kx],d[ky],bins=[arrays[kx+'_edges'],arrays[ky+'_edges']],weights=w)[0]
    if second!=audits:raise ValueError('Input changed between inspection passes')
    for key,total in [('energy_counts',total_weight),('position_weighted_counts',total_weight),
                      ('angle_weighted_counts',total_weight),('projection_weighted_counts',projection_inside_weight),
                      ('weight_counts',count),('radial_divergence_weighted_counts',divergence_inside_weight)]:
        if not np.isclose(arrays[key].sum(),total,rtol=1e-10,atol=1e-12):
            raise ValueError(f'Histogram lost selected records or weights: {key}')
    metadata=dict(species=species,original_histories=histories,available_original_histories=available,
        selected_records=count,selected_nonempty_histories=nonempty,histories_without_selected_species=histories-nonempty,
        photon_divergence=dict(radius_range_mm=[0,80],divergence_range=[0,.5],
            inside_records=divergence_inside_count,outside_records=photon_forward_count-divergence_inside_count,
            inside_weight=divergence_inside_weight,outside_weight=photon_forward_weight-divergence_inside_weight,
            downstream_records=photon_forward_count,excluded_backward_tangent_records=photon_excluded_count,
            statistical_weight_sum=photon_forward_weight,definition='r=hypot(x,y) at recorded plane; tan(theta)=hypot(U,V)/W; theta relative to source +Z',
            normalization='Weighted probability per bin among selected downstream photons; selected species/prefix applies'),
        downstream_records=forward_count,backward_or_tangent_records=count-forward_count,
        statistical_weight_sum=total_weight,statistical_weight_per_original_history=total_weight/histories,
        weighted_mean_energy_MeV=energy_sum/total_weight,source_plane_mm=plane,projection_plane_mm=projection_distance_mm,
        projection_window=dict(x_mm=[-100,100],y_mm=[-100,100],inside_records=projection_inside_count,
            outside_records=forward_count-projection_inside_count,inside_weight=projection_inside_weight,
            outside_weight=forward_weight-projection_inside_weight,
            outside_fraction_of_downstream_weight=(forward_weight-projection_inside_weight)/forward_weight if forward_weight else 0.),
        downstream_statistical_weight_sum=forward_weight,inputs=audits,
        conventions={'coordinates':'source-frame X/Y in mm, Z downstream; not patient LPS',
            'energy':'weighted photon/particle-number density, not energy fluence; normalized by selected total weight',
            'maps':'weighted probability per bin; projection uses downstream particles and selected total weight as denominator',
            'angles':'atan2(U,W) and atan2(V,W), degrees; signed W from particle code',
            'weights':'unweighted record counts',
            'projection':'straight trajectories without hardware, attenuation, or scattering',
            'histories':'ordered original-history prefix including empty histories; all files fully validated',
            'uncertainty':'No independent-particle uncertainty estimates; histories may contain correlated weighted records'})
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=output.name+'.tmp-',dir=output.parent))
    try:
        np.savez_compressed(temp/'histograms.npz',**arrays)
        (temp/'summary.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        render(arrays,metadata,temp/'distributions.png')
        render_divergence(arrays,metadata,temp/'photon_divergence.png')
        if output.exists():raise ValueError('Output appeared during inspection')
        temp.rename(output);temp=None
    finally:
        if temp is not None:shutil.rmtree(temp)
    print(f'{count:,} records; weighted mean energy {metadata["weighted_mean_energy_MeV"]:.6g} MeV')
    print(f'Projection window ±100 mm: {forward_count-projection_inside_count:,} downstream records outside; '
          f'weight {forward_weight-projection_inside_weight:.6g}')
    print(f'Photon divergence: {output/"photon_divergence.png"}')
    print(f'Figure: {output/"distributions.png"}\nSummary: {output/"summary.json"}')
    return output


def render(a,m,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    weight=m['statistical_weight_sum']
    for ax,key,title in [(axes[0,0],'energy','Energy spectrum'),(axes[1,0],'line_energy','Energy detail around 511 keV')]:
        e=a[key+'_edges'];ax.stairs(a[key+'_counts']/weight/np.diff(e),e,color='#2267a5')
        ax.set(title=title,xlabel='Energy (MeV)',ylabel='Weighted probability density (1/MeV)',ylim=(0,None));ax.grid(alpha=.2)
    for ax,name,kx,ky,title,units in [
        (axes[0,1],'position','x','y',f'Recorded plane: Z = {m["source_plane_mm"]:g} mm','mm'),
        (axes[0,2],'angle','angle_x','angle_y','Projected direction angles','deg'),
        (axes[1,1],'projection','projected_x','projected_y',f'Geometric projection: Z = {m["projection_plane_mm"]:g} mm','mm')]:
        image=ax.pcolormesh(a[kx+'_edges'],a[ky+'_edges'],a[name+'_weighted_counts'].T/weight,cmap='inferno',shading='flat')
        ax.set(title=title,xlabel=f'Source-frame X ({units})',ylabel=f'Source-frame Y ({units})',aspect='equal')
        fig.colorbar(image,ax=ax,label='Weighted probability per bin')
        if name=='projection':
            ax.set_xlim(-100,100);ax.set_ylim(-100,100)
            ax.text(.02,.02,f"Outside window: {m['projection_window']['outside_fraction_of_downstream_weight']:.2%} of downstream weight",
                    transform=ax.transAxes,color='white',fontsize=8,
                    bbox={'facecolor':'black','alpha':.6,'edgecolor':'none'})
    ax=axes[1,2];ax.stairs(a['weight_counts'],a['weight_edges'],color='#2267a5')
    ax.set(title='Recorded statistical weights',xlabel='Particle weight',ylabel='Records',ylim=(0,None));ax.grid(alpha=.2)
    fig.suptitle(f'Phase-space inspection: {m["species"]}; {m["selected_records"]:,} records; {m["original_histories"]:,} original histories\n'
        'Energy and maps use statistical weights; projection includes downstream particles only',fontsize=13)
    fig.savefig(path,dpi=160);plt.close(fig)


def render_divergence(a,m,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    info=m['photon_divergence'];weight=info['statistical_weight_sum']
    if weight>0:
        image=ax.pcolormesh(a['radial_edges'],a['divergence_edges'],
            a['radial_divergence_weighted_counts'].T/weight,cmap='inferno',shading='flat')
        fig.colorbar(image,ax=ax,label='Weighted probability per bin')
    else:
        ax.text(.5,.5,'No positive-weight downstream photons in selected species/prefix',
            transform=ax.transAxes,ha='center',wrap=True)
    # Bin the visible window; retain all downstream photon weight in the denominator.
    ax.set(xlim=(0,80),ylim=(0,.5),xlabel='Radial position at phase-space plane (mm)',ylabel='Photon divergence tan(θ)',
        title=f'Photon divergence versus radius at Z = {m["source_plane_mm"]:g} mm\n'
        f'{info["downstream_records"]:,} downstream photons; '
        f'{info["excluded_backward_tangent_records"]:,} backward/tangent photons excluded')
    fig.savefig(path,dpi=160);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('file_bases',nargs='+',help='Ordered IAEA basenames, .header paths or .phsp paths; supply one entry per pair')
    p.add_argument('--output-dir',default='outputs/phase_space_display',help='Fresh output directory')
    p.add_argument('--species',choices=SPECIES,default='photon')
    p.add_argument('--histories',type=int,help='Original-history prefix; default: complete source set. All records are still validated.')
    p.add_argument('--bins',type=int,default=200)
    p.add_argument('--projection-distance-mm',type=float,default=1000.,help='Geometric downstream plane measured from source origin')
    args=p.parse_args(argv)
    try:inspect_sources(args.file_bases,args.output_dir,species=args.species,histories=args.histories,bins=args.bins,projection_distance_mm=args.projection_distance_mm)
    except (ValueError,OSError,KeyError) as e:p.exit(1,f'Error: {e}\n')


if __name__=='__main__':main()
