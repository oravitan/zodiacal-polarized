import os
import numpy as np
import pandas as pd
import healpy as hp
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm
from scipy.optimize import least_squares
from multiprocessing import Pool
from scipy.interpolate import interp1d
from itertools import repeat

from zodipol.utils.paths import BACKGROUND_ISL_FLUX_FILE


BACKGROUND_ISL_FLUX = pd.read_csv(BACKGROUND_ISL_FLUX_FILE, index_col=0)


c = 3e-2  # speed of light, um/s
h = 6.62e-34  # Planck constant, J*s
k = 1.38e-23  # Boltzmann constant, J/K
_2mass_focal_ratio = 13.5  # 2MASS focal ratio


def boltzman(a, T, nu):
    """
    Planck function
    :param a: scaling factor
    :param T: temperature (K)
    :param nu: frequency
    :return: intensity
    """
    return a * 2 * h * nu ** 3 / c ** 2 / (np.exp(h * nu / k / T) - 1)


def get_temp(ii, flux, freq, x0):
    """
    Get the temperature of a pixel, based on the integrated starlight flux
    :param ii: pixel index
    :param flux: integrated starlight flux
    :param freq: frequency
    :param x0: initial guess for the temperature
    :return: temperature (K)
    """
    try:
        flux_ratio = flux[:-1, ii] / (flux[1:, ii] + 1e-20)
        boltzman_ratio = lambda x: boltzman(1, x, freq[:-1]) / (boltzman(1, x, freq[1:]) + 1e-20)
        cost = lambda x: np.sum((flux_ratio - boltzman_ratio(x)) ** 2)
        res_ii = least_squares(cost, x0, bounds=(2500, np.inf))
        return res_ii.x[0]
    except:
        return np.nan


class IntegratedStarlight:
    def __init__(self, isl_map, frequency, catalog=None):
        """
        Integrated starlight model
        :param isl_map: integrated starlight flux data
        :param frequency: frequency
        :param catalog: Vizier catalog
        """
        self.catalog = catalog
        self.isl_map = self._preprocess_map(isl_map)
        self.frequency = frequency

    def save(self, path):
        """
        Save the integrated starlight model
        :param path: path to save the model
        """
        np.savez(path,
                 isl_map=self.isl_map,
                 frequency=self.frequency,
                 isl_units=self.isl_map.unit.to_string())

    @classmethod
    def load(cls, path):
        """
        Load the integrated starlight model
        :param path: path to load the model
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        data = np.load(path)
        units = u.Unit(str(data["isl_units"]))
        return cls(data["isl_map"]*units, data["frequency"])

    @staticmethod
    def _preprocess_map(isl_map):
        """
        Preprocess the integrated starlight map
        :param isl_map: integrated starlight map
        :param frequency: frequency
        :param nside: healpix nside
        :return: preprocessed integrated starlight map
        """
        isl_map = np.nan_to_num(isl_map, nan=0)
        return isl_map

    def interpolate_freq(self, new_frew, update=False):
        """
        Interpolate the integrated starlight flux to a new frequency
        :param new_frew: new frequency
        :return: interpolated integrated starlight flux
        """
        interp_obj = interp1d(self.frequency, self.isl_map)
        interp_skymap = interp_obj(new_frew) * self.isl_map.unit
        if update:
            self.isl_map = interp_skymap
            self.frequency = new_frew
        return interp_skymap

    def resize_skymap(self, nside, update=False):
        """
        Resize the skymap to a new nside
        :param nside: new nside
        :return: resized skymap
        """
        upgraded_map = np.stack([hp.ud_grade(s, nside) for s in self.isl_map.T], axis=-1)
        upgraded_map = upgraded_map * self.isl_map.unit
        if update:
            self.isl_map = upgraded_map
        return upgraded_map


class IntegratedStarlightFactory:
    """
    Integrated starlight model
    """
    def __init__(self, isl_flux=BACKGROUND_ISL_FLUX, catalog="II/246/out", nside=256,
                 focal_ratio=_2mass_focal_ratio):
        """
        Initialize the integrated starlight model
        :param isl_flux: integrated starlight flux data
        :param catalog: Vizier catalog
        :param nside: healpix nside
        """
        self.isl_flux = isl_flux
        self._init_flux_data()

        self.catalog = catalog
        self._reset_visier(columns=["GLON", "GLAT", "Jmag", "eJmag", "Hmag", "eHmag", "Kmag", "eKmag"])
        self.nside = nside
        self.pixels = np.arange(hp.nside2npix(nside))
        self.pixel_size = hp.nside2pixarea(nside, degrees=True) ** 0.5
        self.focal_param = np.pi*(1/(2*focal_ratio))**2 * u.sr

    def _reset_visier(self, columns ="**"):
        """
        Reset the Vizier object
        :param columns: columns to query
        """
        self.visier = Vizier(columns=columns, catalog=self.catalog)
        self.visier.ROW_LIMIT = -1

    def _init_flux_data(self):
        """
        Initialize the flux data
        """
        j_freq, h_freq, k_freq = (self.isl_flux["Lambda"].to_numpy() * u.um).to(u.Hz, equivalencies=u.spectral()).value
        self.freq = np.array((j_freq, h_freq, k_freq))

    def query_direction(self, lon: float, lat: float, width=None, as_dataframe=True):
        """
        Query the Vizier catalog in a given direction
        :param lon: galactic longitude
        :param lat: galactic latitude
        :param width: query width
        :param as_dataframe: return as pandas dataframe
        :return: query result
        """
        width = width if width is not None else self.pixel_size * u.deg  # if not specified, use pixel size
        sky_coord = SkyCoord(lon, lat, unit=(u.deg, u.deg), frame='galactic')
        try:
            result = self.visier.query_region(sky_coord, width=width)
        except:
            return pd.DataFrame()
        result_table = result[0]
        if as_dataframe:
            return result_table.to_pandas()
        return result_table  # only one catalog

    def build_skymap(self, frequency: float, width: u.Quantity = None, show_tqdm: bool = True, parallel: bool = True):
        """
        Build the integrated starlight skymap
        :param frequency: frequency (Hz)
        :param width: query width
        :param show_tqdm: show tqdm progress bar
        :param parallel: use parallel processing
        :return: integrated starlight skymap
        """
        width = width if width is not None else self.pixel_size * u.deg
        lon, lat = self.get_pixels_directions()
        if parallel:
            with Pool() as p:
              flux = p.starmap(self.estimate_direction_flux, tqdm(zip(lon, lat, repeat(frequency), repeat(width)), total=len(lon), disable=not show_tqdm))
        else:
            flux = [self.estimate_direction_flux(lon, lat, frequency, width) for lon, lat in tqdm(zip(lon, lat), total=len(lon), disable=not show_tqdm)]
        flux_px = u.Quantity(flux) / self.pixel_size**2
        isl = IntegratedStarlight(flux_px, frequency)
        return isl

    def process_query(self, result):
        """
        Process the query result
        :param result: query result
        :return: flux
        """
        result = result.dropna()
        j_flux = self.mag2flux(result["Jmag"], "J")
        h_flux = self.mag2flux(result["Hmag"], "H")
        k_flux = self.mag2flux(result["Kmag"], "Ks")
        flux = np.stack((j_flux, h_flux, k_flux))
        return flux

    def get_pixels_directions(self):
        """
        Get the directions of the pixels
        :return: lon, lat
        """
        return hp.pix2ang(self.nside, self.pixels, lonlat=True)

    def estimate_direction_flux(self, lon, lat, frequency, width=None, resample_size=100):
        """
        Estimate the flux in a given direction
        :param lon: galactic longitude
        :param lat: galactic latitude
        :param frequency: frequency (Hz)
        :param width: query width
        :param resample_size: bootstrap resample size
        :return: flux
        """
        # get the flux in the selected direction
        result = self.query_direction(lon, lat, width)
        if len(result) == 0:
            return np.zeros(frequency.shape) * u.Unit('nW / m^2 um sr')
        flux_default_freq = self.process_query(result)

        # bootstrap the flux
        if flux_default_freq.shape[1] > resample_size:
            total_flux = np.sum(flux_default_freq, axis=1)
            resample_ind = np.random.randint(low=0, high=flux_default_freq.shape[1], size=resample_size)
            flux_sample = flux_default_freq[:, resample_ind]
            total_flux_factor = (total_flux / flux_sample.sum(axis=1)).mean()
        else:
            flux_sample = flux_default_freq
            total_flux_factor = 1

        # estimate the temperature and flux factor
        temperature, flux_factor = self._estimate_temperatures(flux_sample, self.freq)
        flux_estimation = boltzman(flux_factor[:, None], temperature[:, None], frequency)
        total_freq_flux = total_flux_factor * flux_estimation.sum(axis=0)
        return total_freq_flux * u.Unit('nW / m^2 um') / self.focal_param / self.pixel_size ** 2

    def mag2flux(self, mag, band):
        """
        Convert magnitude to flux
        :param mag: magnitude
        :param band: frequency band
        :return: flux
        """
        flux_factor = self.isl_flux.loc[band, "Fnu0Mag (W cm-2 um-1)"]
        flux = flux_factor * 10 ** (-mag / 2.5) * 1e4 * 1e9  # nW / m^2 / um
        return flux

    @staticmethod
    def _estimate_temperatures(flux, freq, x0=15000):
        """
        Estimate the temperature of the starlight, using multiprocessing to speed up the process
        :param flux: flux
        :param freq: frequency
        :param x0: initial guess
        :return: temperature, flux factor
        """
        map_inputs = ((r, flux, freq, x0) for r in range(flux.shape[1]))
        res = [get_temp(*args) for args in map_inputs]
        temperature = np.array(res)
        flux_factor = (flux.T / boltzman(1, temperature[:, None], freq)).mean(axis=1)
        return temperature, flux_factor


if __name__ == '__main__':
    wavelength = [300, 400, 500, 600, 700] * u.nm
    frequency = wavelength.to(u.Hz, equivalencies=u.spectral())

    isf = IntegratedStarlightFactory(nside=32)
    skymap_flux = isf.build_skymap(frequency.value, parallel=True)
    skymap_flux.save("saved_models/skymap_flux.npz")

    import matplotlib.pyplot as plt
    import healpy as hp

    hp.mollview(
        skymap_flux.isl_map[:, -1],
        title='Integrated Starlight',
        unit=str(u.nW / u.m ** 2 / u.nm / u.sr),
        cmap="afmhot",
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()


