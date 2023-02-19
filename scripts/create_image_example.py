import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from astropy.time import Time

from zodipol.mie_scattering.mie_scattering_model import MieScatteringModel
from zodipy_local.zodipy_local import Zodipy
from zodipol.imager.imager import Imager
from zodipol.background_radiation.integrated_starlight import IntegratedStarlight
from zodipol.estimation.estimate_signal import estimate_IQU, estimate_DoLP, estimate_AoP

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def generate_spectrum(imager, top_freq=5):
    wavelength_range = imager.get_wavelength_range('red').values * u.nm
    imager_response = imager.get_camera_response(wavelength_range.value, 'red')
    frequency = wavelength_range.to(u.THz, equivalencies=u.spectral())  # Frequency of the observation

    logging.info(f'Getting the top frequencies.')
    top_frequencies = imager_response.argsort()[-top_freq:]  # Top 5 frequencies

    frequency = frequency[top_frequencies]  # Select the top 5 frequencies
    wavelength = wavelength_range[top_frequencies]  # Select the top 5 wavelengths
    imager_response = imager_response[top_frequencies]  # Select the top 5 imager responses

    # sort by frequency
    frequency_ord = np.argsort(frequency)
    frequency = frequency[frequency_ord]
    wavelength = wavelength[frequency_ord]
    imager_response = imager_response[frequency_ord]
    return wavelength, frequency, imager_response


if __name__ == '__main__':
    # set params
    logging.info(f'Started run.')
    nside = 128  # Healpix resolution
    polarizance = 1  # Polarizance of the observation
    fov = 5  # deg
    polarization_angle = np.linspace(0, np.pi, 60, endpoint=False)  # Polarization angle of the observation

    # Initialize the model
    logging.info(f'Initializing the model.')
    imager = Imager()
    model = Zodipy("dirbe", solar_cut=30 * u.deg, extrapolate=True, parallel=True)  # Initialize the model

    # Generate the spectrum
    logging.info(f'Getting the wavelength range.')
    wavelength, frequency, imager_response = generate_spectrum(imager, top_freq=5)
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
        isl_n_electrons = imager.intensity_to_number_of_electrons(isl.isl_map[..., None], wavelength=wavelength, weights=imager_response)
    else:
        isl_n_electrons = 0

    # Calculate the emission at pixels
    logging.info(f'Getting the binned emission.')
    binned_emission = model.get_binned_emission_pix(
        frequency,
        weights=frequency_weight,
        pixels=np.arange(hp.nside2npix(nside)),
        nside=nside,
        obs_time=Time("2022-06-14"),
        obs="earth",
        polarization_angle=polarization_angle,
        polarizance=polarizance,
        mie_scattering_model=mie_model)

    # Calculate the polarization
    logging.info(f'Calculating the polarization.')
    I, Q, U = estimate_IQU(binned_emission, polarization_angle)
    binned_dolp = estimate_DoLP(I, Q, U)
    binned_aop = estimate_AoP(Q, U)

    # Plot the emission of the first polarization angle
    logging.info(f'Plotting the emission of the first polarization angle.')
    for ii in np.linspace(0, binned_emission.shape[-1], 4, endpoint=False, dtype=int):
        hp.mollview(
            binned_emission[..., 0, ii],
            title="Binned zodiacal emission at {} with polarization angle {}".format(wavelength[0],np.round(polarization_angle[ii], 2)),
            unit=str(binned_emission.unit),
            min=0,
            cmap="afmhot",
            rot=(0, 0, 0)
        )
        hp.graticule()
        plt.show()

    # plot the binned polarization
    logging.info(f'Plotting the binned polarization.')
    hp.mollview(
        binned_dolp[..., 0],
        title="Binned zodiacal polarization at {}".format(wavelength[-1]),
        unit="MJy/sr",
        cmap="afmhot",
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()

    hp.mollview(
        binned_aop[..., 0],
        title="Binned angle of polarization at {}".format(wavelength[-1]),
        unit="MJy/sr",
        cmap="afmhot",
        min=-np.pi,
        max=np.pi,
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()

    # Calculate the number of photons
    logging.info(f'Calculating the realistic image with noise.')
    n_electrons = imager.intensity_to_number_of_electrons(binned_emission, frequency=frequency, weights=imager_response)
    n_electrons = n_electrons + isl_n_electrons
    n_electrons_noised = imager.imager_noise_model(n_electrons)
    camera_intensity = imager.number_of_electrons_to_intensity(n_electrons_noised, frequency, imager_response)

    I, Q, U = estimate_IQU(camera_intensity, polarization_angle)
    camera_dolp = estimate_DoLP(I, Q, U)
    camera_aop = estimate_AoP(Q, U)

    # Plot the emission of the first polarization angle
    logging.info(f'Plotting the camera intensity of the first polarization angle.')
    for ii in np.linspace(0, camera_intensity.shape[-1], 4, endpoint=False, dtype=int):
        hp.mollview(
            camera_intensity[..., ii],
            title="Camera Intensity with polarization angle {}".format(np.round(polarization_angle[ii], 2)),
            unit=str(camera_intensity.unit),
            min=0,
            cmap="afmhot",
            rot=(0, 0, 0)
        )
        hp.graticule()
        plt.show()

    logging.info(f'Plotting the camera polarization.')
    hp.mollview(
        camera_dolp,
        title="Camera polarization",
        unit="MJy/sr",
        cmap="afmhot",
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()

    hp.mollview(
        camera_aop,
        title="Camera angle of polarization",
        unit="MJy/sr",
        cmap="afmhot",
        min=-np.pi,
        max=np.pi,
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()
