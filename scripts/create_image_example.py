import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import logging
from astropy.time import Time

from zodipy_local.zodipy_local import Zodipy
from zodipol.imager.imager import Imager

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=logging_format)


if __name__ == '__main__':
    # set params
    nside = 256  # Healpix resolution
    central_wavelength = 0.62 * u.um  # Wavelength of the observation
    polarizance = 1  # Polarizance of the observation
    fov = 5  # deg
    polarization_angle = np.linspace(0, np.pi, 60, endpoint=False)  # Polarization angle of the observation
    logging.debug(f'Started run.')

    # Initialize the model
    logging.debug(f'Initializing the model.')
    imager = Imager()
    model = Zodipy("dirbe", solar_cut=30 * u.deg, extrapolate=True, parallel=False)  # Initialize the model

    logging.debug(f'Getting the wavelength range.')
    wavelength_range = imager.get_wavelength_range('red').values * u.nm
    imager_response = imager.get_camera_response(wavelength_range.value, 'red')
    frequency = wavelength_range.to(u.THz, equivalencies=u.spectral())  # Frequency of the observation

    logging.debug(f'Getting the top frequencies.')
    top_frequencies = imager_response.argsort()[-5:]  # Top 5 frequencies
    frequency = frequency[top_frequencies]  # Select the top 5 frequencies
    wavelength = wavelength_range[top_frequencies]  # Select the top 5 wavelengths
    imager_response = imager_response[top_frequencies]  # Select the top 5 imager responses

    # Calculate the emission at pixels
    frequency_ord = np.sort(frequency)
    frequency_weight = np.ones_like(frequency_ord)

    logging.debug(f'Getting the binned emission.')
    binned_emission = model.get_binned_emission_pix(
        frequency_ord,
        weights=frequency_ord,
        pixels=np.arange(hp.nside2npix(nside)),
        nside=nside,
        obs_time=Time("2022-06-14"),
        obs="earth",
        polarization_angle=polarization_angle,
        polarizance=polarizance)

    # Calculate the polarization
    logging.debug(f'Calculating the polarization.')
    emission_max, emission_min = np.max(binned_emission, axis=-1), np.min(binned_emission, axis=-1)
    binned_polarization = (emission_max - emission_min) / (emission_max + emission_min)

    # Plot the emission of the first polarization angle
    logging.debug(f'Plotting the emission of the first polarization angle.')
    for ii in np.linspace(0, binned_emission.shape[-1], 4, endpoint=False, dtype=int):
        hp.mollview(
            binned_emission[..., 0, ii],
            title="Binned zodiacal emission at {} with polarization angle {}".format(wavelength[0],np.round(polarization_angle[ii], 2)),
            unit=str(binned_emission.unit),
            norm='log',
            cmap="afmhot",
            rot=(0, 0, 0)
        )
        hp.graticule()
        plt.show()

    # plot the binned polarization
    logging.debug(f'Plotting the binned polarization.')
    hp.mollview(
        binned_polarization[:, 0],
        title="Binned zodiacal polarization at {}".format(wavelength[-1]),
        unit="MJy/sr",
        cmap="afmhot",
        min=0,
        max=1,
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()

    # Calculate the number of photons
    logging.debug(f'Calculating the realistic image with noise.')
    n_electrons = imager.intensity_to_number_of_electrons(binned_emission, frequency, imager_response)
    n_electrons_noised = imager.imager_noise_model(n_electrons)
    camera_intensity = imager.number_of_electrons_to_intensity(n_electrons_noised, frequency, imager_response)

    # Plot the emission of the first polarization angle
    logging.debug(f'Plotting the emission of the first polarization angle.')
    for ii in np.linspace(0, camera_intensity.shape[-1], 4, endpoint=False, dtype=int):
        hp.mollview(
            camera_intensity[..., ii],
            title="Binned zodiacal emission at {} with polarization angle {}".format(central_wavelength, np.round(polarization_angle[ii], 2)),
            unit=str(camera_intensity.unit),
            min=0,
            cmap="afmhot",
            rot=(0, 0, 0)
        )
        hp.graticule()
        plt.show()

