"""Native photon steering with exact, target-independent square-field tiling."""
import numpy as np
from pyRadPlan.stf.generators import StfGeneratorPhotonIMRT, register_generator


class StfGeneratorPhotonSquareField(StfGeneratorPhotonIMRT):
    """Tile a square at isocenter; native code constructs all rays and beamlets."""
    name = 'Photon square-field beamlets'
    short_name = 'photonSquareField'
    field_width_mm = None

    def _generate_ray_positions_in_isocenter_plane(self, beam):
        width, bixel = self.field_width_mm, beam['bixel_width']
        for name, value in [('field_width_mm', width), ('bixel_width', bixel)]:
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a positive finite number')
        count = int(round(width / bixel))
        if count < 1 or not np.isclose(count * bixel, width, rtol=1e-10, atol=1e-9):
            raise ValueError('Square field width must be an integer multiple of bixel width')
        centers = (np.arange(count) + .5) * bixel - width / 2
        x, z = np.meshgrid(centers, centers, indexing='ij')
        return np.vstack((x.ravel(), np.zeros(count * count), z.ravel()))


register_generator(StfGeneratorPhotonSquareField)
