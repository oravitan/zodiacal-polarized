import numpy as np
import astropy.units as u
from astropy.modeling.models import BlackBody
from zodipy import Zodipy

from utils.math import normalize


wl_spectrum = np.logspace(np.log10(300), np.log10(700), 10) * u.nm  # white light wavelength in nm
cobe_factor = 6.073e-5  # COBE irradiance factor


def get_solar_irradiance(spectrum: np.ndarray) -> np.ndarray:
    """
    Get the solar irradiance spectrum.
    :param spectrum: wavelength in nm
    :return: solar irradiance in MJy/sr
    """
    solar_black_body = BlackBody(5778 * u.K)  # solar black body model
    solar_irradiance = solar_black_body(spectrum * u.nm).to('MJy/sr').value * cobe_factor  # solar irradiance in MJy/sr COBE
    return solar_irradiance


def get_solar_probability_density(solar_irradiance: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """
    Get the solar probability density spectrum.
    :param solar_irradiance: solar irradiance in MJy/sr
    :param spectrum: wavelength in nm
    :return: solar probability density
    """
    return normalize(solar_irradiance * np.gradient(spectrum))


def get_dirbe_solar_irradiance() -> (np.ndarray, np.ndarray):
    """
    Get the solar irradiance spectrum from DIRBE.
    :return: solar irradiance in MJy/sr
    """
    model = Zodipy("dirbe", solar_cut=30 * u.deg, extrapolate=True)  # Initialize the model
    spectrum = model.ipd_model.spectrum.to('nm').value  # wavelength in nm
    solar_irradiance = model.ipd_model.solar_irradiance  # solar irradiance in MJy/sr
    return spectrum, solar_irradiance


if __name__ == '__main__':
    import matplotlib.pyplot as plt  # import here to avoid unnecessary import
    spectrum = wl_spectrum.to('nm').value  # wavelength in nm
    solar_irradiance = get_solar_irradiance(spectrum)  # solar irradiance in MJy/sr
    # spectrum, solar_irradiance = get_dirbe_solar_irradiance()
    solar_probability_density = get_solar_probability_density(solar_irradiance, spectrum)

    # plot the solar intensity spectrum
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.loglog(spectrum, solar_irradiance, 'b', ls='dashdot', lw=1, label="Solar Intensity Spectrum")
    ax2.set_xlabel(r"Wavelength (nm)")
    ax2.set_ylabel(r"Intensity (MJy/sr)")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()
