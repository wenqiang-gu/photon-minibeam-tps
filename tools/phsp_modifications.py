"""Numerics for the standalone IAEA modifier; no transport or field selection."""
import numpy as np
from numpy.polynomial.legendre import leggauss

BASE = (1.120813595915993, 4.174177344588501, -1.276750337246975)
PRESETS = {'baseline': BASE,
    'peak_minus_0p1': (0.662660721290608, 4.136222597137568, -0.9067781743433319),
    'peak_plus_0p1': (0.9823102326993479, 5.408931306994932, -1.6902537833662665)}
LINE = .51099895
PROTECTION = .0005
VERSION = 'model13-focal-spot-v2'
LOW_ENERGY_KEEP = .1
LOW_ENERGY_FULL = .2

def transition(energy):
    t=np.clip((np.asarray(energy,dtype=float)-LOW_ENERGY_KEEP)/(LOW_ENERGY_FULL-LOW_ENERGY_KEEP),0,1)
    return t*t*(3-2*t)



def log_continuum(e, p):
    """Ali & Rogers 2012 model 13 continuum photon density, before normalization."""
    e = np.asarray(e, dtype=float)
    if np.any(~np.isfinite(e) | (e < .01) | (e > 5.75)):
        raise ValueError('Spectrum modification requires photon energies in 0.01–5.75 MeV')
    W = np.exp(np.polynomial.polynomial.polyval(e, [6.575,-36.23,-1.578]) /
        np.polynomial.polynomial.polyval(e, [1,9.667,.7132,-.0003778]))
    l = np.log(e)
    Al = np.polynomial.polynomial.polyval(l, [.06107,.01694,-.002390,.0003116,.0005286,.00002507]) / np.polynomial.polynomial.polyval(l, [1,.7769,.2434,.03838,.003042,.00009686])
    return (np.log(1+p[2]*e/5.75+(e/5.75)**2)
        + np.log(np.log(5.75*(5.75-e)/e+1.65)-.5)-np.log(e)-W*p[0]**2-Al*p[1]**2)


def log_integral(p):
    edges = np.r_[.01, np.arange(.025, 5.75, .025), 5.75]
    g,w = leggauss(32)
    e = edges[:-1,None]+np.diff(edges)[:,None]*(g+1)/2
    z = log_continuum(e,p); maximum = z.max()
    return float(maximum+np.log(np.sum(np.exp(z-maximum)*w*np.diff(edges)[:,None]/2)))


class Spectrum:
    def __init__(self, parameters):
        self.p = tuple(parameters)
        self.identity = self.p == BASE
        self.norm = log_integral(BASE)-log_integral(self.p)

    def ratio(self, energy, correction=1.):
        if self.identity:
            return np.ones(len(energy))
        with np.errstate(over='ignore', under='ignore'):
            t=transition(energy)
            r = np.exp(t*(log_continuum(energy,self.p)-log_continuum(energy,BASE)+self.norm))
            # Taper the fluence correction too: a constant correction would
            # introduce a weight discontinuity at the protected lower boundary.
            r *= 1+(correction-1)*t
        if np.any(~np.isfinite(r) | (r <= 0)):
            raise ValueError('Spectrum multiplier overflow or underflow')
        return r


def photon_masks(a):
    photon = np.abs(a['code'].astype(int)) == 1
    e=np.abs(a['energy'].astype(float))
    protected = photon & ((e <= LOW_ENERGY_KEEP) | (np.abs(e-LINE) <= PROTECTION))
    return photon, protected, photon & ~protected


def gaussian_pairs(ids, seed):
    """Counter-based SplitMix64/Box–Muller; invariant to chunking and variants."""
    def uniform(salt):
        with np.errstate(over='ignore'):
            z = ids.astype(np.uint64) + np.uint64(seed) + np.uint64(salt)
            z = (z ^ (z >> 30))*np.uint64(0xbf58476d1ce4e5b9)
            z = (z ^ (z >> 27))*np.uint64(0x94d049bb133111eb)
            z ^= z >> 31
        return ((z >> 12).astype(float)+.5)/2**52
    u = uniform(0x9e3779b97f4a7c15);v = uniform(0x3c6ef372fe94f82a)
    radius = np.sqrt(-2*np.log(u))
    return radius*np.cos(2*np.pi*v), radius*np.sin(2*np.pi*v)


def spread(a, ids, sx, sy, distance, seed):
    """Rotate downstream photons; preserve position and angular residual."""
    if sx == sy == 0:
        return 0
    u=a['u'].astype(float);v=a['v'].astype(float)
    use=(a['code']==1)&(u*u+v*v<1)
    if not use.any():return 0
    gx,gy=gaussian_pairs(ids[use],seed)
    nominal=np.column_stack((a['x'][use].astype(float)*10,a['y'][use].astype(float)*10,np.full(use.sum(),distance)))
    displaced=nominal.copy();displaced[:,0]-=gx*sx;displaced[:,1]-=gy*sy
    nominal/=np.linalg.norm(nominal,axis=1)[:,None]
    displacement_norm=np.linalg.norm(displaced,axis=1)
    if np.any(~np.isfinite(displaced)) or np.any(~np.isfinite(displacement_norm)):
        raise ValueError('Focal-spot displacement overflow')
    displaced/=displacement_norm[:,None]
    k=np.cross(nominal,displaced);cos=np.sum(nominal*displaced,axis=1)
    if np.any(cos <= -1+1e-12):raise ValueError('Antiparallel focal-spot transformation')
    d=np.column_stack((u[use],v[use],np.sqrt(1-u[use]**2-v[use]**2)))
    out=d+np.cross(k,d)+np.cross(k,np.cross(k,d))/(1+cos[:,None])
    norms=np.linalg.norm(out,axis=1)
    if np.any(~np.isfinite(out)) or np.any(~np.isfinite(norms) | (norms<=0)):
        raise ValueError('Nonfinite focal-spot transformation')
    out/=norms[:,None]
    # Float32 rounding must not turn a forward direction into a tangent one.
    uv=out[:,:2].astype(np.float32)
    bad=(out[:,2]<=0)|(np.sum(uv.astype(float)**2,axis=1)>=1)
    good=np.flatnonzero(use)[~bad]
    a['u'][good]=uv[~bad,0];a['v'][good]=uv[~bad,1]
    return int(bad.sum())
