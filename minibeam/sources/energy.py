"""Discrete empirical spectrum and an injectable monoenergetic test utility."""
import numpy as np
from ..topas.contracts import ParameterFragment

class MonoenergeticEnergy:
    def sample(self, context, rng, count):
        return np.full(count, self.render(context).record['energy_mev'])

    def render(self, context):
        energy = float(context.beamlet.energy)
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError("Photon energy must be positive")
        return ParameterFragment(
            f'd:So/Beam/BeamEnergy = {energy:.12g} MeV\nu:So/Beam/BeamEnergySpread = 0',
            record={"energy_mev": energy})


class EmpiricalSpectrum:
    """Discrete sampling of the supplied energies; independent of steering energy."""
    def __init__(self, energies=None, weights=None):
        from .empirical import EMPIRICAL_ENERGY_MEV, EMPIRICAL_SPECTRUM_WEIGHT
        self.energies = np.array(EMPIRICAL_ENERGY_MEV if energies is None else energies, dtype=float, copy=True)
        self.weights = np.array(EMPIRICAL_SPECTRUM_WEIGHT if weights is None else weights, dtype=float, copy=True)

    def sample(self, context, rng, count):
        record = self.render(context).record
        return rng.choice(record['energy_values_mev'], size=count, p=record['spectrum_probabilities'])

    def render(self, context):
        energies, weights = self.energies, self.weights
        if (energies.ndim != 1 or energies.size == 0 or weights.shape != energies.shape
                or not np.isfinite(energies).all() or not np.isfinite(weights).all()
                or np.any(energies <= 0) or np.any(np.diff(energies) <= 0)
                or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0):
            raise ValueError("Spectrum requires increasing positive energies and matching finite nonnegative weights with positive sum")
        probabilities = weights / weights.sum()
        values = ' '.join(f'{x:.17g}' for x in energies)
        probs = ' '.join(f'{x:.17g}' for x in probabilities)
        return ParameterFragment(
            's:So/Beam/BeamEnergySpectrumType = "Discrete"\n'
            f'dv:So/Beam/BeamEnergySpectrumValues = {energies.size} {values} MeV\n'
            f'uv:So/Beam/BeamEnergySpectrumWeights = {energies.size} {probs}',
            record={"energy_model": "discrete_spectrum", "energy_values_mev": energies.tolist(),
                    "spectrum_weights": weights.tolist(), "spectrum_probabilities": probabilities.tolist()})
