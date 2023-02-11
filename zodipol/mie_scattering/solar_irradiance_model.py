import numpy as np
import astropy.units as u
from astropy.modeling.models import BlackBody


def get_solar_irradiance(spectrum: np.ndarray) -> np.ndarray:
    """
    Get the solar irradiance spectrum.
    :param spectrum: wavelength in nm
    :return: solar irradiance in MJy/sr
    """
    solar_black_body = BlackBody(5778 * u.K)  # solar black body model
    solar_irradiance = solar_black_body(spectrum * u.nm).to('MJy/sr').value  # solar irradiance in MJy/sr COBE
    return solar_irradiance

