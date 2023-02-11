import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy_local.zodipy_local import Zodipy
from zodipol.camera.camera_trasformations import intensity_to_number_of_electrons, number_of_electrons_to_intensity
from zodipol.camera.camera_noise import complete_noise_model


def get_satellite_response(central_frequency, frequency_std, frequency_n_samples):
    """
    Get the camera response.
    :param central_frequency: central frequency in THz
    :param frequency_std: frequency standard deviation in THz
    :param frequency_n_samples: number of frequency samples
    :return: camera response
    """
    frequency = u.Quantity(np.linspace(central_frequency - 2 * frequency_std, central_frequency + 2 * frequency_std, frequency_n_samples))  # Frequency range
    frequency_weight = np.exp(-((frequency - central_frequency) / frequency_std) ** 2)  # Frequency weights
    frequency_weight /= np.sum(frequency_weight)  # Normalize the frequency weights
    return frequency, frequency_weight


if __name__ == '__main__':
    # set params
    nside = 256  # Healpix resolution
    central_wavelength = 1 * u.um  # Wavelength of the observation
    polarizance = 1  # Polarizance of the observation
    polarization_angle = np.linspace(0, np.pi, 60, endpoint=False)  # Polarization angle of the observation

    # Initialize the model
    model = Zodipy("dirbe", solar_cut=30 * u.deg, extrapolate=True, parallel=True)  # Initialize the model
    central_frequency = central_wavelength.to(u.THz, equivalencies=u.spectral())  # Frequency of the observation
    frequency, frequency_weight = get_satellite_response(central_frequency, 1 * u.THz, 10)

    # Calculate the emission at pixels
    binned_emission = model.get_binned_emission_pix(
        frequency,
        weights=frequency_weight,
        pixels=np.arange(hp.nside2npix(nside)),
        nside=nside,
        obs_time=Time("2022-06-14"),
        obs="earth",
        polarization_angle=polarization_angle,
        polarizance=polarizance,
    )

    # Calculate the polarization
    emission_max, emission_min = np.max(binned_emission, axis=1), np.min(binned_emission, axis=1)
    binned_polarization = (emission_max - emission_min) / (emission_max + emission_min)

    # Plot the emission of the first polarization angle
    hp.mollview(
        binned_emission[..., 0],
        title="Binned zodiacal emission at {} with polarization angle {}".format(central_wavelength,
                                                                                 np.round(polarization_angle[0], 2)),
        unit=str(binned_emission.unit),
        min=0,
        cmap="afmhot",
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()

    # plot the binned polarization
    hp.mollview(
        binned_polarization,
        title="Binned zodiacal polarization at {}".format(central_wavelength),
        unit="MJy/sr",
        cmap="afmhot",
        min=0,
        max=1,
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()

    # Calculate the number of photons
    n_electrons = intensity_to_number_of_electrons(binned_emission, frequency, frequency_weight)
    n_electrons_noised = complete_noise_model(n_electrons)
    camera_intensity = number_of_electrons_to_intensity(n_electrons_noised, frequency, frequency_weight)

    # Plot the emission of the first polarization angle
    hp.mollview(
        camera_intensity[..., 0],
        title="Binned zodiacal emission at {} with polarization angle {}".format(central_wavelength, np.round(polarization_angle[0], 2)),
        unit=str(camera_intensity.unit),
        min=0,
        cmap="afmhot",
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()

