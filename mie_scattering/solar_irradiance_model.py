import numpy as np
import astropy.units as u
from astropy.modeling.models import BlackBody

from utils.math import normalize


cobe_factor = 6.073e-5  # COBE irradiance factor


class SolarIrradianceModel:
    def __init__(self, spectrum=None, n_samples: int = 10):
        """
        Initialize the solar irradiance model.
        :param spectrum: wavelength in nm
        :param n_samples: number of samples
        """
        self.spectrum = spectrum.to('nm').value  # wavelength in nm
        if spectrum is None:
            self.spectrum = self._get_wl_spectrum(n_samples).to('nm').value  # wavelength in nm
        self.solar_irradiance = self._get_solar_irradiance(self.spectrum)  # solar irradiance in MJy/sr
        self.solar_likelihood = normalize(self.solar_irradiance * np.gradient(self.spectrum))  # normalized solar irradiance

    @staticmethod
    def _get_wl_spectrum(n_samples: int = 10) -> u.Quantity:
        """
        Get the white light spectrum.
        :param n_samples: number of samples
        :return: wavelength in nm
        """
        wl_spectrum = np.logspace(np.log10(300), np.log10(700), n_samples) * u.nm  # white light wavelength in nm
        return wl_spectrum

    @staticmethod
    def _get_solar_irradiance(spectrum: np.ndarray) -> np.ndarray:
        """
        Get the solar irradiance spectrum.
        :param spectrum: wavelength in nm
        :return: solar irradiance in MJy/sr
        """
        solar_black_body = BlackBody(5778 * u.K)  # solar black body model
        solar_irradiance = solar_black_body(spectrum * u.nm).to('MJy/sr').value * cobe_factor  # solar irradiance in MJy/sr COBE
        return solar_irradiance


if __name__ == '__main__':
    import matplotlib.pyplot as plt  # import here to avoid unnecessary import

    spectrum = np.logspace(np.log10(1300), np.log10(3000), 20)
    sim = SolarIrradianceModel(spectrum)

    # plot the solar intensity spectrum
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.loglog(sim.spectrum, sim.solar_irradiance, 'b', ls='dashdot', lw=1, label="Solar Intensity Spectrum")
    ax2.set_xlabel(r"Wavelength (nm)")
    ax2.set_ylabel(r"Intensity (MJy/sr)")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()
