import numpy as np
from multiprocessing import Pool
from PyMieScatt import MieS1S2
from functools import cache

from mie_scattering.plotting import plot_mueller_matrix_elems, plot_intensity_polarization
from mie_scattering.particle_size_model import ParticleSizeModel
from mie_scattering.solar_irradiance_model import SolarIrradianceModel
from utils.constants import refractive_ind
from utils.math import normalize


class MieScatteringModel:
    def __init__(self, wavelength, refractive_index_dict=None, particle_size: ParticleSizeModel = None, theta_res=361,
                 wavelength_weight=None):
        if refractive_index_dict is None:
            refractive_index_dict = refractive_ind
        if particle_size is None:
            particle_size = ParticleSizeModel()
        wavelength = np.array(wavelength)  # wavelength in um
        if wavelength_weight is None:
            wavelength_weight = self._get_black_body_weights(wavelength)

        self.refractive_index = list(refractive_index_dict.keys())
        self.refractive_index_weight = np.array(list(refractive_index_dict.values()))

        self.particle_size = particle_size
        self.wavelength = wavelength

        self.theta = np.linspace(0, np.pi, theta_res)  # scattering angle in rad
        self.mu = np.cos(self.theta)  # cosine of scattering angle

        self.x = np.pi * self.particle_size.particle_size[..., None] / wavelength[None, ...]  # Mie parameter
        self.x_weights = normalize(particle_size.particle_likelihood[..., None] * wavelength_weight[None, ...])  # Mie parameter weights

        self.S1, self.S2 = self._calculate_S1S2()

        self.SL, self.SR, self.SU = self._calculate_SLSRSU(self.S1, self.S2)
        self.P = (self.SL - self.SR) / (self.SL + self.SR)
        self.S11, self.S12, self.S33, self.S34 = self._calculate_mueller_elems(self.S1, self.S2)

    def get_mueller_matrix(self, theta):
        S11, S12, S33, S34 = self.get_mie_scattering_mueller_elem(theta)
        return self.get_mueller_matrix_from_elem(S11, S12, S33, S34)

    @staticmethod
    def get_cross_section_norm(theta: np.ndarray, S11: np.ndarray):
        return np.trapz(S11, theta) / np.pi

    def get_mueller_matrix_from_elem(self, S11: np.ndarray, S12: np.ndarray, S33: np.ndarray, S34: np.ndarray):
        """
        Get the Mueller matrix
        :param S11: S11 element
        :param S12: S12 element
        :param S33: S33 element
        :param S34: S34 element
        :return: Mueller matrix
        """
        assert len(S11) == len(S12) == len(S33) == len(S34), 'S11, S12, S33, S34 must have the same length'
        cross_section_norm = self.get_cross_section_norm(self.theta, self.S11)
        M = np.zeros((len(S11), 4, 4), dtype=np.float64)
        M[:, 0, 0] = S11
        M[:, 0, 1] = S12
        M[:, 1, 0] = S12
        M[:, 1, 1] = S11

        M[:, 2, 2] = S33
        M[:, 2, 3] = -S34
        M[:, 3, 2] = S34
        M[:, 3, 3] = S33
        M /= cross_section_norm
        return M

    def get_mie_scattering_mueller_elem(self, theta: np.ndarray):
        """
        Get the mie scattering mueller matrix elements for some theta
        :param theta: scattering angle in rad
        :return: S11, S12, S33, S34
        """
        S11 = np.interp(theta, self.theta, self.S11)
        S12 = np.interp(theta, self.theta, self.S12)
        S33 = np.interp(theta, self.theta, self.S33)
        S34 = np.interp(theta, self.theta, self.S34)
        return S11, S12, S33, S34

    @staticmethod
    def _calculate_SLSRSU(S1, S2):
        SL, SR = np.real(S1.conj() * S1), np.real(S2.conj() * S2)
        SU = (SL + SR) / 2
        return SL, SR, SU

    def _calculate_mueller_elems(self, S1, S2):
        S11 = 0.5 * (np.abs(S2) ** 2 + np.abs(S1) ** 2).real
        S12 = 0.5 * (np.abs(S2) ** 2 - np.abs(S1) ** 2).real
        S33 = 0.5 * (np.conjugate(S2) * S1 + S2 * np.conjugate(S1)).real
        S34 = (0.5j * (S1 * np.conjugate(S2) - S2 * np.conjugate(S1))).real
        return S11, S12, S33, S34

    def _calculate_S1S2(self):
        """
        Calculate the scattering function for a given particle size distribution, spectrum, and scattering angle.
        :return: scattering function (size param x cos scattering angle x refractive index x S1/S2)
        """
        S1, S2 = self._get_scattering_function(self.refractive_index, self.x.ravel(), self.mu)
        S1_avg = np.sum((S1 * self.refractive_index_weight).sum(axis=-1) * self.x_weights.ravel()[..., None], axis=0)
        S2_avg = np.sum((S2 * self.refractive_index_weight).sum(axis=-1) * self.x_weights.ravel()[..., None], axis=0)
        return S1_avg, S2_avg

    @staticmethod
    def _get_scattering_function(m: list, x_list: list | np.ndarray, mu_list: list) -> (np.ndarray, np.ndarray):
        """
        Get the scattering function for a given refractive index, size parameter, and scattering angle,
        Using synced parallel processing to speed up the calculation.
        :param m: refractive index
        :param x_list: list of size parameter
        :param mu_list: list of cosine of scattering angle
        :return: scattering function (size param x cos scattering angle x refractive index x S1/S2)
        """
        with Pool() as p:
            scatt = p.starmap(MieS1S2, [(mm, x, mu) for x in x_list for mu in mu_list for mm in m])
        scatt_resh = np.array(scatt).reshape((len(x_list), len(mu_list), len(m), 2))
        return scatt_resh[..., 0], scatt_resh[..., 1]  # S1, S2

    def _get_black_body_weights(self, wavelength) -> np.ndarray:
        """
        Get the black body weights for the given wavelength
        :param wavelength: wavelength in nm
        :return: black body weights
        """
        solar_weights = SolarIrradianceModel._get_solar_irradiance(wavelength)
        return normalize(solar_weights)


if __name__ == '__main__':
    wavelength = 1000
    psm = ParticleSizeModel()
    mie_scatt = MieScatteringModel(wavelength, refractive_ind, psm)

    # plot the Mueller matrix elements
    plot_mueller_matrix_elems(mie_scatt.theta, mie_scatt.S11, mie_scatt.S12, mie_scatt.S33, mie_scatt.S34)
    plot_intensity_polarization(mie_scatt.theta, mie_scatt.SL, mie_scatt.SR, mie_scatt.SU, mie_scatt.P)

    # plot the Mueller matrix
    # S11, S12, S33, S34 = mie_scatt.get_mie_scattering_mueller_elem(mie_scatt.theta)
    # M = mie_scatt.get_mueller_matrix_from_elem(S11, S12, S33, S34)
    # print(M[..., 0])
    # print(M[..., 1])
    # print(M[..., 2])
