import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy_local.zodipy import Zodipy


def get_satellite_response(central_frequency, frequency_std, frequency_n_samples):
    """
    Get the satellite response.
    :param central_frequency: central frequency in THz
    :param frequency_std: frequency standard deviation in THz
    :param frequency_n_samples: number of frequency samples
    :return: satellite response
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
        title="Binned zodiacal emission at {} with polarization angle {}".format(central_wavelength, np.round(polarization_angle[0], 2)),
        unit="MJy/sr",
        min=0,
        max=1,
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

