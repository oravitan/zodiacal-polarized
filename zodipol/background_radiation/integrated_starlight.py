import logging
import os
import numpy as np
import pandas as pd
import healpy as hp
from astroquery.vizier import Vizier
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm
from scipy.optimize import least_squares
from multiprocessing import Pool, cpu_count
from scipy.interpolate import interp1d
from itertools import repeat
from retry import retry
from requests import ConnectTimeout, ConnectionError

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
    def __init__(self, isl_map, frequency, catalog=None, obs_time: Time = None):
        """
        Integrated starlight model
        :param isl_map: integrated starlight flux data
        :param frequency: frequency
        :param catalog: Vizier catalog
        """
        self.catalog = catalog
        self.isl_map = self._preprocess_map(isl_map)
        self.frequency = frequency
        self.obs_time = obs_time

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

    def get_ang(self, theta, phi):
        """
        Get the integrated starlight flux at a given angle
        :param theta: theta angle
        :param phi: phi angle
        :return: integrated starlight flux
        """
        nside = hp.npix2nside(self.isl_map.shape[0])
        return self.isl_map[hp.ang2pix(nside, theta, phi), ...]

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
        current_size = hp.npix2nside(self.isl_map.shape[0])  # current nside
        res_factor = hp.nside2pixarea(current_size) / hp.nside2pixarea(nside)  # resolution factor
        upgraded_map = np.stack([hp.ud_grade(s.value, nside, power=-2) for s in self.isl_map.T], axis=-1)  # rescale the skymap
        upgraded_map = res_factor * upgraded_map * self.isl_map.unit
        if update:  # update the skymap
            self.isl_map = upgraded_map
        return upgraded_map


class IntegratedStarlightFactory:
    """
    Integrated starlight model
    """
    def __init__(self, obs_time, isl_flux=BACKGROUND_ISL_FLUX, catalog="II/246/out", nside=256,
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
        self.obs_time = Time(obs_time)

    def _reset_visier(self, columns ="**"):
        """
        Reset the Vizier object
        :param columns: columns to query
        """
        self.visier = Vizier(columns=columns, catalog=self.catalog)
        self.visier.cache_location = '/dev/null'
        self.visier.ROW_LIMIT = 30000  # -1

    def _init_flux_data(self):
        """
        Initialize the flux data
        """
        j_freq, h_freq, k_freq = (self.isl_flux["Lambda"].to_numpy() * u.um).to(u.Hz, equivalencies=u.spectral()).value
        self.freq = np.array((j_freq, h_freq, k_freq))

    @retry(ConnectTimeout, delay=5, jitter=5, logger=logging.getLogger(__name__))
    @retry(ConnectionError, delay=5, jitter=5, logger=logging.getLogger(__name__))
    def query_direction(self, lon, lat, width=None, as_dataframe=True, request_id=None):
        """
        Query the Vizier catalog in a given direction
        :param lon: galactic longitude
        :param lat: galactic latitude
        :param width: query width
        :param as_dataframe: return as pandas dataframe
        :return: query result
        """
        if request_id is not None:
            logging.info(f"Querying Vizier catalog {self.catalog} for request {request_id}")
        width = width if width is not None else self.pixel_size * u.deg  # if not specified, use pixel size
        sky_coord = SkyCoord(lon, lat, unit=(u.deg, u.deg), frame='geocentricmeanecliptic', obstime=self.obs_time)

        for ii in range(3):
            result = self.visier.query_region(sky_coord, width=width, cache=False)  # cache=False
            if len(result) > 0:
                break
            logging.warning(f"Vizier catalog {self.catalog} returned emptry result for {request_id}")

        if len(result) == 0:  # after retrying
            logging.warning(f"Empty Vizier catalog {self.catalog} query result for request {request_id}")
            return pd.DataFrame({'_q': []})

        result_table = result[0]
        if as_dataframe:
            return result_table.to_pandas()
        return result_table  # only one catalog

    def build_skymap(self, frequency: float, width: u.Quantity = None, parallel: bool = True, request_size=1000):
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
        return self.build_dirmap(lon, lat, frequency=frequency, width=width, parallel=parallel, request_size=request_size)

    def build_dirmap(self, lon, lat, frequency: float, width: u.Quantity = None,
                     parallel: bool = True, request_size=1000):
        logging.info(f'Building ISL skymap for {len(lon)} pixels.')
        array_split = np.array_split(np.stack([lon, lat], axis=-1), len(lon) // request_size)
        if parallel:
            n_cpu = cpu_count() - 1
            logging.info(f'Using {len(array_split)} requests parallel in {n_cpu} pools.')
            with Pool(30) as p:
                flux_list = p.starmap(self.estimate_direction_flux_parallel, zip(array_split, range(len(array_split)),
                                                                                 repeat(frequency), repeat(width)))
                flux = np.concatenate(flux_list) * flux_list[0][0].unit
        else:
            logging.info(f'Using {len(array_split)} requests.')
            flux = [self.estimate_direction_flux_parallel(lonlat, id, frequency, width) for id,lonlat in enumerate(array_split)]
        isl = IntegratedStarlight(u.Quantity(flux), frequency)
        return isl

    def estimate_direction_flux_parallel(self, lonlat, request_id, frequency, width):
        lon, lat = lonlat[..., 0], lonlat[..., 1]
        query_res = self.query_direction(lon, lat, width, request_id=request_id)
        res = []
        for ii in tqdm(range(1, len(lon)+1), desc=f"Request {request_id}"):
            res.append(self.estimate_direction_flux(query_res[query_res._q == ii], frequency, width))
        return res

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

    def estimate_direction_flux(self, query_result, frequency, width, resample_size=100):
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
        if len(query_result) == 0:
            return np.zeros(frequency.shape) * u.Unit('nW / m^2 um sr')
        flux_default_freq = self.process_query(query_result)

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
        return total_freq_flux * u.Unit('nW / m^2 um') / self.focal_param / width.to('deg').value ** 2

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

    isf = IntegratedStarlightFactory(obs_time="2022-06-14", nside=256)
    skymap_flux = isf.build_skymap(frequency.value, parallel=True)
    skymap_flux.save("saved_models/skymap_flux.npz")

    import matplotlib.pyplot as plt
    import healpy as hp

    hp.mollview(
        skymap_flux.isl_map[:, -1],
        title='Integrated Starlight',
        unit=str(skymap_flux.isl_map.unit),
        cmap="afmhot",
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()


