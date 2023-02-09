import numpy as np
from multiprocessing import Pool
from PyMieScatt import MieS1S2

from mie_scattering.plotting import plot_mueller_matrix_elems, plot_intensity_polarization
from mie_scattering.particle_size_model import ParticleSizeModel
from utils.constants import refractive_ind


class MieScatteringModel:
    def __init__(self, refractive_index_dict, particle_size: ParticleSizeModel, wavelength, theta_res=361):
        self.refractive_index = list(refractive_index_dict.keys())
        self.refractive_index_weight = np.array(list(refractive_index_dict.values()))

        self.particle_size = particle_size
        self.wavelength = wavelength

        self.theta = np.linspace(0, np.pi, theta_res)  # scattering angle in rad
        self.mu = np.cos(self.theta)  # cosine of scattering angle

        self.x = np.pi * self.particle_size.particle_size / wavelength
        self.S1, self.S2 = self._calculate_S1S2()

        self.SL, self.SR, self.SU = self._calculate_SLSRSU(self.S1, self.S2)
        self.P = (self.SL - self.SR) / (self.SL + self.SR)
        self.S11, self.S12, self.S33, self.S34 = self._calculate_mueller_elems(self.S1, self.S2)

    @staticmethod
    def get_mueller_matrix_from_elem(S11: np.ndarray, S12: np.ndarray, S33: np.ndarray, S34: np.ndarray):
        """
        Get the Mueller matrix
        :param S11: S11 element
        :param S12: S12 element
        :param S33: S33 element
        :param S34: S34 element
        :return: Mueller matrix
        """
        assert len(S11) == len(S12) == len(S33) == len(S34), 'S11, S12, S33, S34 must have the same length'
        M = np.zeros((len(S11), 4, 4), dtype=np.float64)
        M[:, 0, 0] = S11
        M[:, 0, 1] = S12
        M[:, 1, 0] = S12
        M[:, 1, 1] = S11

        M[:, 2, 2] = S33
        M[:, 2, 3] = -S34
        M[:, 3, 2] = S34
        M[:, 3, 3] = S33
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
        S11 = 0.5 * (np.real(S2.conj() * S2) + np.real(S1.conj() * S1))
        S12 = 0.5 * (np.real(S2.conj() * S2) - np.real(S1.conj() * S1))
        S33 = np.real(0.5 * (S2.conj() * S1 + S1.conj() * S2))
        S34 = np.real(1j * 0.5 * (S2.conj() * S1 - S1.conj() * S2))
        return S11, S12, S33, S34

    def _calculate_S1S2(self):
        """
        Calculate the scattering function for a given particle size distribution, spectrum, and scattering angle.
        :return: scattering function (size param x cos scattering angle x refractive index x S1/S2)
        """
        S1, S2 = self._get_scattering_function(self.refractive_index, self.x, self.mu)
        S1_avg = np.sum((S1 * self.refractive_index_weight).sum(axis=-1) * self.particle_size.particle_likelihood[..., None], axis=0)
        S2_avg = np.sum((S2 * self.refractive_index_weight).sum(axis=-1) * self.particle_size.particle_likelihood[..., None], axis=0)
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


if __name__ == '__main__':
    wavelength = 1300
    psm = ParticleSizeModel()
    mie_scatt = MieScatteringModel(refractive_ind, psm, wavelength)

    # plot the Mueller matrix elements
    plot_mueller_matrix_elems(mie_scatt.theta, mie_scatt.S11, mie_scatt.S12, mie_scatt.S33, mie_scatt.S34)
    plot_intensity_polarization(mie_scatt.theta, mie_scatt.SL, mie_scatt.SR, mie_scatt.SU, mie_scatt.P)

    # plot the Mueller matrix
    S11, S12, S33, S34 = mie_scatt.get_mie_scattering_mueller_elem(np.array([0.1, 0.2, 0.3]))
    M = mie_scatt.get_mueller_matrix_from_elem(S11, S12, S33, S34)
    print(M[..., 0])
    print(M[..., 1])
    print(M[..., 2])
