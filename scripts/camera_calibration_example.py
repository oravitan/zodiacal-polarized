import astropy.units as u
import healpy as hp
import numpy as np
import logging
import os
from astropy.time import Time
from scipy.optimize import least_squares

from zodipol.mie_scattering.mie_scattering_model import MieScatteringModel
from zodipy_local.zodipy_local import Zodipy
from zodipol.imager.imager import Imager
from zodipol.background_radiation.integrated_starlight import IntegratedStarlight
from scripts.create_image_example import generate_spectrum

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def make_camera_images(I, U, Q, polarizance, polarization_angle, imager, model, frequency, imager_response, add_noise=True):
    binned_emission_real = model.IUQ_to_image(I, U, Q, polarizance, polarization_angle)

    # Calculate the number of photons
    n_electrons_real = imager.intensity_to_number_of_electrons(binned_emission_real, frequency=frequency, weights=imager_response)
    if add_noise:
        n_electrons_real = imager.imager_noise_model(n_electrons_real)
    else:
        n_electrons_real += imager.camera_dark_current_estimation()
        n_electrons_real = imager.camera_post_process(n_electrons_real)
    camera_intensity_real = imager.number_of_electrons_to_intensity(n_electrons_real, frequency, imager_response)
    return camera_intensity_real


if __name__ == '__main__':
    # set params
    logging.info(f'Started run.')
    nside = 64  # Healpix resolution
    polarizance = 1  # Polarizance of the observation
    fov = 5  # deg
    polarization_shift = 0.1
    polarization_angle = polarization_shift + np.linspace(0, np.pi, 4, endpoint=False)  # Polarization angle of the observation
    polarization_angle_diff = np.diff(polarization_angle)[0]

    # Initialize the model
    logging.info(f'Initializing the model.')
    imager = Imager()
    model = Zodipy("dirbe", solar_cut=30 * u.deg, extrapolate=True, parallel=True)  # Initialize the model

    # Generate the spectrum
    logging.info(f'Getting the wavelength range.')
    wavelength, frequency, imager_response = generate_spectrum(imager, n_freq=20)
    frequency_weight = np.ones_like(frequency)  # Weight of the frequencies

    # Load the mie model
    mie_model_path = 'saved_models/white_light_mie_model.npz'
    if os.path.isfile(mie_model_path):
        logging.info(f'Loading the mie model: {mie_model_path}')
        mie_model = MieScatteringModel.load(mie_model_path)
    else:
        spectrum = np.logspace(np.log10(300), np.log10(700), 20)  # white light wavelength in nm
        mie_model = MieScatteringModel.train(spectrum)

    # Load the integrated starlight model
    integrated_starlight_path = 'saved_models/skymap_flux.npz'
    if os.path.isfile(integrated_starlight_path):
        logging.info(f'Loading the integrated starlight model: {integrated_starlight_path}')
        isl = IntegratedStarlight.load(integrated_starlight_path)
        isl.resize_skymap(nside, update=True)
        isl.interpolate_freq(frequency.to('Hz'), update=True)

        df, dw = np.gradient(frequency), -np.gradient(wavelength)
        isl_map = isl.isl_map * (dw / df)[None, ...]
    else:
        isl_map = 0

    # Calculate the emission at pixels
    logging.info(f'Getting the binned emission.')
    binned_emission = model.get_binned_emission_pix(
        frequency,
        weights=frequency_weight,
        pixels=np.arange(hp.nside2npix(nside)),
        nside=nside,
        obs_time=Time("2022-06-14"),
        obs="earth",
        mie_scattering_model=mie_model,
        return_IQU=True
    )
    binned_emission = np.nan_to_num(binned_emission, nan=0)
    I, U, Q = binned_emission[..., 0], binned_emission[..., 1], binned_emission[..., 2]
    I = I + isl_map

    logging.info(f'Create real images.')
    df_ind = polarization_angle_diff * np.arange(binned_emission.shape[-1])
    make_img = lambda eta: make_camera_images(I, U, Q, polarizance, eta+df_ind, imager, model, frequency, imager_response, add_noise=False)

    polarization_angle_res = []
    for ii in range(100):
        camera_intensity_real = make_camera_images(I, U, Q, polarizance, polarization_angle, imager, model, frequency, imager_response)
        cost_function = lambda eta: 1e23 * (make_img(eta) - camera_intensity_real).value.flatten()
        res = least_squares(cost_function, x0=0, bounds=(-np.pi/2, np.pi/2))
        polarization_angle_res.append(res.x)
    pass

