import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from tqdm import tqdm

from zodipy_local.zodipy_local import Zodipy
from zodipol.imager.imager import Imager


if __name__ == '__main__':
    # set params
    nside = 256  # Healpix resolution
    central_wavelength = 0.62 * u.um  # Wavelength of the observation
    polarizance = 1  # Polarizance of the observation
    polarization_angle = np.linspace(0, np.pi, 60, endpoint=False)  # Polarization angle of the observation

    # Initialize the model
    imager = Imager()
    model = Zodipy("dirbe", solar_cut=30 * u.deg, extrapolate=True, parallel=True)  # Initialize the model

    wavelength_range = imager.get_wavelength_range('red').values * u.nm
    imager_response = imager.get_camera_response(wavelength_range.value, 'red')
    frequency = wavelength_range.to(u.THz, equivalencies=u.spectral())  # Frequency of the observation

    top_frequencies = imager_response.argsort()[-5:]  # Top 5 frequencies
    frequency = frequency[top_frequencies]  # Select the top 5 frequencies
    wavelength = wavelength_range[top_frequencies]  # Select the top 5 wavelengths
    imager_response = imager_response[top_frequencies]  # Select the top 5 imager responses

    # frequency, frequency_weight = get_satellite_response(central_frequency, 1 * u.THz, 10)

    # Calculate the emission at pixels
    binned_emission = np.stack([
        model.get_binned_emission_pix(
        f,
        pixels=np.arange(hp.nside2npix(nside)),
        nside=nside,
        obs_time=Time("2022-06-14"),
        obs="earth",
        polarization_angle=polarization_angle,
        polarizance=polarizance)
        for f in tqdm(frequency)
    ], axis=-1)

    # Calculate the polarization
    emission_max, emission_min = np.max(binned_emission, axis=1), np.min(binned_emission, axis=1)
    binned_polarization = (emission_max - emission_min) / (emission_max + emission_min)

    # Plot the emission of the first polarization angle
    for ii in np.linspace(0, binned_emission.shape[-2], 4, endpoint=False, dtype=int):
        hp.mollview(
            binned_emission[..., ii, -1],
            title="Binned zodiacal emission at {} with polarization angle {}".format(wavelength[-1],np.round(polarization_angle[ii], 2)),
            unit=str(binned_emission.unit),
            norm='log',
            cmap="afmhot",
            rot=(0, 0, 0)
        )
        hp.graticule()
        plt.show()

    # plot the binned polarization
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
    n_electrons = imager.intensity_to_number_of_electrons(binned_emission, frequency, imager_response)
    n_electrons_noised = imager.imager_noise_model(n_electrons)
    camera_intensity = imager.number_of_electrons_to_intensity(n_electrons_noised, frequency, imager_response)

    # Plot the emission of the first polarization angle
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

