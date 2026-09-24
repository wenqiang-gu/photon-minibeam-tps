"""User-supplied discrete photon spectrum; energies in MeV, relative weights.

Values are preserved as supplied. This is not a scanner/linac calibration.
"""
import numpy as np

EMPIRICAL_ENERGY_MEV = np.asarray((
    0.13, 0.38, 0.63, 0.88, 1.13, 1.38, 1.63, 1.88,
    2.13, 2.38, 2.63, 2.88, 3.13, 3.38, 3.63, 3.88,
    4.13, 4.38, 4.63, 4.88, 5.13, 5.38, 5.63, 5.88,
    6.13, 6.38, 6.63, 6.88,
), dtype=np.float64)
EMPIRICAL_SPECTRUM_WEIGHT = np.asarray((
    0.032047941, 0.139322874, 0.128271317, 0.10410132,
    0.087722697, 0.074515256, 0.061865442, 0.051816519,
    0.044850118, 0.038812754, 0.033491917, 0.029783373,
    0.025720295, 0.022598543, 0.019733182, 0.017636211,
    0.015519179, 0.014236603, 0.012188171, 0.010440358,
    0.009278173, 0.007876917, 0.006674784, 0.005250251,
    0.003518869, 0.001849387, 0.000746224, 0.00013132,
), dtype=np.float64)
EMPIRICAL_ENERGY_MEV.setflags(write=False)
EMPIRICAL_SPECTRUM_WEIGHT.setflags(write=False)
