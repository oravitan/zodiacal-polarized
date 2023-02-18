import numpy as np
from astropy import units as u

from zodipol.background_radiation.integrated_starlight import IntegratedStarlight


if __name__ == '__main__':
    wavelength = [300, 400, 500, 600, 700] * u.nm
    frequency = wavelength.to(u.Hz, equivalencies=u.spectral())

    isf = IntegratedStarlightFactory(nside=16)
    skymap_flux = isf.build_skymap(frequency.value, parallel=True)
    skymap_flux.save("saved_models/skymap_flux.npz")
