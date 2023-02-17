import numpy as np
import pandas as pd
import healpy as hp
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm
from scipy.optimize import least_squares

from zodipol.utils.paths import BACKGROUND_ISL_FLUX_FILE


BACKGROUND_ISL_FLUX = pd.read_csv(BACKGROUND_ISL_FLUX_FILE, index_col=0)


c = 3e-2  # speed of light, um/s
h = 6.62e-34  # Planck constant, J*s
k = 1.38e-23  # Boltzmann constant, J/K
boltzman = lambda a, T, nu: a * 2 * h * nu ** 3 / c ** 2 / (np.exp(h * nu / k / T) - 1)  # Planck function


class IntegratedStarlight:
    def __init__(self, isl_flux=BACKGROUND_ISL_FLUX, catalog="II/246/out", nside=256):
        self.isl_flux = isl_flux
        self._init_flux_data()

        self.catalog = catalog
        self._reset_visier(columns=["GLON", "GLAT", "Jmag", "eJmag", "Hmag", "eHmag", "Kmag", "eKmag"])
        self.nside = nside
        self.pixels = np.arange(hp.nside2npix(nside))
        self.pixel_size = hp.nside2pixarea(256, degrees=True) ** 0.5

    def _reset_visier(self, columns: str | list ="**"):
        self.visier = Vizier(columns=columns, catalog=self.catalog)
        self.visier.ROW_LIMIT = -1

    def _init_flux_data(self):
        j_freq, h_freq, k_freq = (self.isl_flux["Lambda"].to_numpy() * u.um).to(u.Hz, equivalencies=u.spectral()).value
        self.freq = np.array((j_freq, h_freq, k_freq))

    def query_direction(self, ra: float, dec: float, width=None, as_dataframe=True):
        width = width or self.pixel_size * u.deg  # if not specified, use pixel size
        sky_coord = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='galactic')
        result = self.visier.query_region(sky_coord, width=width)
        result_table = result[0]
        if as_dataframe:
            return result_table.to_pandas()
        return result_table  # only one catalog

    def process_query(self, result):
        j_flux = self.mag2flux(result["Jmag"], "J")
        h_flux = self.mag2flux(result["Hmag"], "H")
        k_flux = self.mag2flux(result["Kmag"], "Ks")
        flux = np.stack((j_flux, h_flux, k_flux))
        return flux

    def get_pixels_directions(self):
        return hp.pix2ang(self.nside, self.pixels, lonlat=True)

    def estimate_direction_flux(self, ra, dec, frequency, width=None, resample_size=2000):
        result = self.query_direction(ra, dec, width)
        flux_default_freq = self.process_query(result)
        total_flux = np.sum(flux_default_freq, axis=1)

        resample_ind = np.random.randint(low=0, high=flux_default_freq.shape[1], size=resample_size)
        flux_sample = flux_default_freq[:, resample_ind]
        total_flux_factor = (total_flux / flux_sample.sum(axis=1)).mean()

        temperature, flux_factor = self._estimate_temperatures(flux_sample, self.freq, show_tqdm=True)
        flux_estimation = boltzman(flux_factor[:, None], temperature[:, None], frequency)
        total_freq_flux = total_flux_factor * flux_estimation.sum(axis=0)
        return total_freq_flux * u.nW / u.m**2 / u.nm

    def mag2flux(self, mag, band):
        flux_factor = self.isl_flux.loc[band, "Fnu0Mag (W cm-2 um-1)"]
        flux = flux_factor * 10 ** (-mag / 2.5) * 1e4 * 1e9  # nW / m^2 / um
        return flux

    @staticmethod
    def _estimate_temperatures(flux, freq, x0=15000, show_tqdm=False):
        res = []
        for ii in tqdm(range(flux.shape[1]), disable=not show_tqdm):
            flux_ratio = flux[:-1, ii] / (flux[1:, ii] + 1e-20)
            boltzman_ratio = lambda x: boltzman(1, x, freq[:-1]) / (boltzman(1, x, freq[1:]) + 1e-20)
            cost = lambda x: np.sum((flux_ratio - boltzman_ratio(x)) ** 2)
            res_ii = least_squares(cost, x0, bounds=(0, np.inf))
            res.append(res_ii.x[0])
        temperature = np.array(res)
        flux_factor = (flux.T / boltzman(1, temperature[:, None], freq)).mean(axis=1)
        return temperature, flux_factor


if __name__ == '__main__':
    wavelength = [300, 400, 500, 600, 700] * u.nm
    frequency = wavelength.to(u.Hz, equivalencies=u.spectral())

    isl = IntegratedStarlight()
    pixel_lon, pixel_lat = isl.get_pixels_directions()
    result = isl.estimate_direction_flux(0, 0, frequency.value)
    print(result)
    pass
    # to get number of electrons
    # energy_factor = imager.exposure_time * imager.pixel_area * imager.optical_loss
    # (result * energy_factor * d_w / (h * frequency)).si
