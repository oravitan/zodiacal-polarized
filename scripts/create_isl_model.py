import numpy as np
import logging
from astropy import units as u

from zodipol.background_radiation.integrated_starlight import IntegratedStarlightFactory
from zodipol.visualization.skymap_plots import plot_skymap


logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


if __name__ == '__main__':
    logging.info('Started ISL calculation run.')
    wavelength = [300, 400, 500, 600, 700] * u.nm
    frequency = wavelength.to(u.Hz, equivalencies=u.spectral())

    isf = IntegratedStarlightFactory('2022-06-14', nside=512)
    skymap_flux = isf.build_skymap(frequency.value, parallel=True, request_size=10000, n_cpu=30)
    skymap_flux.save("saved_models/skymap_flux512.npz")

    plot_skymap(skymap_flux.isl_map[:, 0], max=1e4)
