"""Small PDF spectrum audit from streaming, statistical-weight histograms."""
import numpy as np
from tools.phsp_modifications import BASE, Spectrum, log_continuum, log_integral, continuum_peak, spectrum_for_variant


def write_spectrum_report(folder,variants,edges,before,after,corrections,particle_types,electron_input=0.0,electron_output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    x=np.unique(np.r_[np.linspace(.01,5.75,3000),.51049895,.51099895,.51149895])
    with PdfPages(folder/'spectrum_report.pdf') as pdf:
        for name,v in variants.items():
            p=v['parameters'];model=spectrum_for_variant(v)
            fig,axes=plt.subplots(2,2,figsize=(11.7,8.3))
            if v['spectrum']=='alpha':
                baseline=np.exp(log_continuum(x,BASE)-log_integral(BASE))
                ratio=model.ratio(x,corrections[name]);ratio[(x<=.1)|(abs(x-.51099895)<=.0005)]=1
                axes[0,0].plot(x,baseline,label='Baseline model')
                axes[0,0].plot(x,baseline*ratio,label='Tilted model (empirical correction)')
            else:
                for params,label in [(BASE,'Baseline model'),(p,'Requested model')]:
                    axes[0,0].plot(x,np.exp(log_continuum(x,params)-log_integral(params)),label=label)
            axes[0,0].set(xlim=(0,1.5),title='Smooth model continuum (unit area)',ylabel='Density (1/MeV)')
            axes[0,0].legend(fontsize=8)
            for hist,label,style in [(after[name],'Written photons','-'),(before,'Input retained photons','--')]:
                # Common denominator preserves amplitude comparison and the line.
                if before.sum()>0:
                    density=hist/(before.sum()*np.diff(edges))
                    for ax in (axes[0,1],axes[1,0]):ax.stairs(density,edges,label=label,lw=1.2,linestyle=style)
            axes[0,1].set(xlim=(0,5.75),title='Actual weighted spectrum: full energy range',ylabel='Weight density / input in-range weight')
            axes[1,0].set(xlim=(0,1.2),title='Actual spectrum close-up (includes 511 keV)',ylabel='Weight density / input in-range weight')
            for ax in (axes[0,1],axes[1,0]):
                if before.sum()>0:ax.legend(fontsize=8)
                else:ax.text(.5,.5,'No selected photons',transform=ax.transAxes,ha='center')
            ratio=model.ratio(x,corrections[name]);ratio[(x<=.1)|(abs(x-.51099895)<=.0005)]=1
            axes[1,1].plot(x,ratio)
            axes[1,1].set(title='Applied photon-weight multiplier',ylabel='Output / input weight',yscale='log')
            for ax in axes.flat:ax.set_xlabel('Photon energy (MeV)');ax.grid(alpha=.2)
            peak=continuum_peak(p)
            fig.suptitle(f'{name}: spectrum audit\nModel continuum peak {peak:.6f} MeV; C1={p[0]:.7g}, C2={p[1]:.7g}, C3={p[2]:.7g}',fontsize=13)
            if v['spectrum']=='alpha':
                fig.suptitle(f"{name}: exponential spectrum tilt\nalpha = {v['alpha_per_mev']:g} MeV⁻¹; 511 keV photon weights unchanged",fontsize=13)
                axes[0,0].set_title('Model illustration; normalization from actual particles')
            fit=v.get('peak_fit',{})
            notes=(f'Species: {", ".join(particle_types)}. Endpoint 5.75 MeV. Actual histograms: 10 keV bins, all listed parts.\n'
                   f'Request: {fit.get("requested_setting",v["spectrum"])} = {fit.get("requested_value",v.get('alpha_per_mev',"preset/custom"))}; '
                   f'continuum correction {corrections[name]:.7g}.\n'
                   'Peak is the smooth model mode, not a guarantee for the empirical output histogram. No particle energies are shifted.\n'
                   'Low-energy photons and the 511 keV band retain weights; original histories unchanged. Fit is a local optimum, not a calibration.\n'
                   'Baseline/custom inputs outside 0–5.75 MeV are omitted from histograms; nonbaseline energies outside model support are rejected.')
            scale=v.get('electron_weight_scale',1.0)
            written=(electron_output or {}).get(name,0.0)
            notes+=(f'\nIncident electron scale: {scale:g}; retained weight {electron_input:.8g} -> {written:.8g}. '
                    'No mixed-species renormalization; secondary electrons are not scaled.')
            if 'electron' not in particle_types:notes+=' Electrons excluded: no effect.'
            fig.text(.06,.025,notes,fontsize=8,va='bottom')
            fig.tight_layout(rect=(0,.18,1,.91));pdf.savefig(fig)
            fig.savefig(folder/name/'spectrum_report.png',dpi=140);plt.close(fig)
            np.savetxt(folder/name/'spectrum_histogram.csv',np.column_stack((edges[:-1],edges[1:],before,after[name])),
                       delimiter=',',header='energy_low_MeV,energy_high_MeV,input_weight,output_weight',comments='')
