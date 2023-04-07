import logging
import numpy as np
import healpy as hp
from astropy.time import Time
import matplotlib.pyplot as plt

from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol, Observation
from zodipol.visualization.skymap_plots import plot_satellite_image, plot_satellite_image_indices

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


if __name__ == '__main__':
    logging.info(f'Started run.')
    parser = ArgParser()

    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"],
                      planetary=parser["planetary"], isl=parser["isl"], resolution=parser["resolution"],
                      imager_params=parser["imager_params"], mie_model_path='saved_models/wide_mie_model.npz')

    nside = 64
    pixels = np.arange(hp.nside2npix(nside))
    optical_depth = zodipol.zodipy.get_binned_optical_depth_pix(zodipol.frequency, pixels, nside, obs_time=Time("2022-06-14"),
                                                mie_scattering_model=zodipol.mie_model, weights=zodipol.frequency_weight)
    pass
