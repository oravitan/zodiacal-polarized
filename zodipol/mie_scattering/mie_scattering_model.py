import numpy as np
import os
from multiprocessing import Pool
from PyMieScatt import MieS1S2, MieQ
from scipy.ndimage import convolve
from scipy.interpolate import RegularGridInterpolator, interp1d
from itertools import repeat

from zodipol.visualization.mie_plotting import plot_mueller_matrix_elems, plot_intensity_polarization
from zodipol.mie_scattering.particle_size_model import ParticleTable, DEFAULT_PARTICLE_MODEL
from zodipol.utils.constants import refractive_ind


class MieScatteringModel:
    # ------------------  model training  ------------------
    def __init__(self, wavelength, theta, S1, S2, extinction=None):
        self.wavelength = np.array(wavelength)  # wavelength in um
        self.theta = theta  # scattering angle in rad
        self.mu = np.cos(self.theta)  # cosine of scattering angle
        self.S1, self.S2 = S1, S2  # Mie scattering amplitudes
        self.S1_grid = RegularGridInterpolator((self.wavelength, self.theta), self.S1)  # Mie scattering amplitude interpolator
        self.S2_grid = RegularGridInterpolator((self.wavelength, self.theta), self.S2)  # Mie scattering amplitude interpolator
        self.extinction = extinction  # extinction cross section

    @classmethod
    def train(cls, wavelength, particle_table: ParticleTable = None, theta_res=361, kernel_size=20):
        if particle_table is None:
            particle_table = DEFAULT_PARTICLE_MODEL
        wavelength = np.array(wavelength)  # wavelength in nm

        theta = np.linspace(0, np.pi, theta_res)  # scattering angle in rad
        mu = np.cos(theta)  # cosine of scattering angle
        S1, S2 = cls._calculate_S1S2(wavelength, particle_table, mu, kernel_size=kernel_size)

        q_ext, q_sca, q_abs = cls._calculate_extinction(wavelength, particle_table)
        extinction = {'q_ext': q_ext, 'q_sca': q_sca, 'q_abs': q_abs}
        return cls(wavelength, theta, S1, S2, extinction=extinction)

    # ------------------  model save and load  ------------------
    def save(self, filename):
        extinction = np.stack(list(self.extinction.values()), axis=-1)
        np.savez(filename, wavelength=self.wavelength, theta=self.theta, S1=self.S1, S2=self.S2, extinction=extinction)

    @classmethod
    def load(cls, filename):
        data = np.load(filename, allow_pickle=True)
        extinction = {'q_ext': data['extinction'][:, 0], 'q_sca': data['extinction'][:, 1], 'q_abs': data['extinction'][:, 2]}
        return cls(data['wavelength'], data['theta'], data['S1'], data['S2'], extinction=extinction)

    # ------------------  model prediction  ------------------
    def __call__(self, *args, **kwargs):
        return self.get_mueller_matrix(*args, **kwargs)

    def get_mueller_matrix(self, wavelength, theta):
        wavelength_m, theta_m = np.meshgrid(wavelength, theta)
        S1, S2 = self._interp_S1S2(wavelength_m, theta_m)
        S11, S12, S33, S34 = self._calculate_mueller_elems(S1, S2)
        cross_section = self._get_cross_section(wavelength)
        return self.get_mueller_matrix_from_elem(S11, S12, S33, S34, cross_section=cross_section)

    def get_extinction(self, wavelength):
        q_ext, q_sca, q_abs = self.extinction['q_ext'], self.extinction['q_sca'], self.extinction['q_abs']
        q_ext_w = interp1d(self.wavelength, q_ext, kind='linear', fill_value='extrapolate')(wavelength)
        q_sca_w = interp1d(self.wavelength, q_sca, kind='linear', fill_value='extrapolate')(wavelength)
        q_abs_w = interp1d(self.wavelength, q_abs, kind='linear', fill_value='extrapolate')(wavelength)
        return q_ext_w, q_sca_w, q_abs_w

    @staticmethod
    def _calculate_extinction(wavelength, particle_table):
        q_ext, q_sca, q_abs = 0, 0, 0
        for particle, particle_prc in particle_table.get_model_list():
            refractive_index = particle.refractive_index
            particle_size = particle.particle_size
            cur_ext_res = MieScatteringModel._get_extinction_coeff(refractive_index, wavelength, particle_size)
            cur_q_ext, cur_q_sca, cur_q_abs = cur_ext_res[..., 0], cur_ext_res[..., 1], cur_ext_res[..., 2]
            q_ext += particle_prc * np.sum(cur_q_ext * particle.particle_likelihood[None, :], axis=-1)
            q_sca += particle_prc * np.sum(cur_q_sca * particle.particle_likelihood[None, :], axis=-1)
            q_abs += particle_prc * np.sum(cur_q_abs * particle.particle_likelihood[None, :], axis=-1)
        return q_ext, q_sca, q_abs

    def get_scattering(self, wavelength, theta):
        wavelength_m, theta_m = np.meshgrid(wavelength, theta)
        S1, S2 = self._interp_S1S2(wavelength_m, theta_m)
        SL, SR, SU = self._calculate_SLSRSU(S1, S2)
        P = self._calculate_polarization(SL, SR)
        return SL, SR, SU, P

    def get_mueller_matrix_from_elem(self, S11: np.ndarray, S12: np.ndarray, S33: np.ndarray, S34: np.ndarray,
                                     cross_section: float = 1):
        """
        Get the Mueller matrix
        :param S11: S11 element
        :param S12: S12 element
        :param S33: S33 element
        :param S34: S34 element
        :param cross_section: cross section of the particle
        :return: Mueller matrix
        """
        assert len(S11) == len(S12) == len(S33) == len(S34), 'S11, S12, S33, S34 must have the same length'
        M = np.zeros((S11.shape + (4, 4)), dtype=np.float64)
        M[..., 0, 0] = S11
        M[..., 0, 1] = S12
        M[..., 1, 0] = S12
        M[..., 1, 1] = S11

        M[..., 2, 2] = S33
        M[..., 2, 3] = -S34
        M[..., 3, 2] = S34
        M[..., 3, 3] = S33
        M /= cross_section[..., None, None]
        return M

    def _get_cross_section(self, wavelength):
        theta = np.linspace(0, np.pi, 180, endpoint=True)
        wavelength, theta = np.meshgrid(wavelength, theta)
        S1, S2 = self._interp_S1S2(wavelength, theta)
        S11 = 0.5 * (np.abs(S2) ** 2 + np.abs(S1) ** 2).real
        return 2*np.pi*np.trapz(S11 * np.sin(theta), theta, axis=0)[None, ...]

    # ------------------ Internal Methods ------------------
    def _interp_S1S2(self, wavelength: float, theta):
        S1 = self.S1_grid((wavelength, theta))
        S2 = self.S2_grid((wavelength, theta))
        return S1, S2

    @staticmethod
    def _calculate_SLSRSU(S1, S2):
        SL, SR = np.real(S1.conj() * S1), np.real(S2.conj() * S2)
        SU = (SL + SR) / 2
        return SL, SR, SU

    @staticmethod
    def _calculate_polarization(SL, SR):
        return (SL - SR) / (SL + SR)

    @staticmethod
    def _calculate_mueller_elems(S1, S2) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        S11 = 0.5 * (np.abs(S2) ** 2 + np.abs(S1) ** 2).real
        S12 = 0.5 * (np.abs(S2) ** 2 - np.abs(S1) ** 2).real
        S33 = 0.5 * (np.conjugate(S2) * S1 + S2 * np.conjugate(S1)).real
        S34 = (0.5j * (S1 * np.conjugate(S2) - S2 * np.conjugate(S1))).real
        return S11, S12, S33, S34

    @staticmethod
    def _calculate_S1S2(wavelength, particle_table: ParticleTable, mu, kernel_size: int = None):
        """
        Calculate the scattering function for a given particle size distribution, spectrum, and scattering angle.
        :return: scattering function (size param x cos scattering angle x refractive index x S1/S2)
        """
        S1, S2 = 0, 0
        for particle, particle_prc in particle_table.get_model_list():
            refractive_index = particle.refractive_index
            x = np.pi * particle.particle_size[..., None] / wavelength[None, ...]
            cur_S1, cur_S2 = MieScatteringModel._get_scattering_function(refractive_index, x, mu)

            S1 = S1 + particle_prc * np.sum(cur_S1 * particle.particle_likelihood[..., None, None], axis=0)
            S2 = S2 + particle_prc * np.sum(cur_S2 * particle.particle_likelihood[..., None, None], axis=0)

        if kernel_size is not None:  # smooth S1, S2 with a moving average
            kernel = np.ones((1, kernel_size)) / kernel_size
            S1 = convolve(S1, kernel, mode='nearest')
            S2 = convolve(S2, kernel, mode='nearest')
        return S1, S2

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
        x_flat = x_list.reshape((-1))
        with Pool() as p:
            scatt = p.starmap(MieS1S2, [(m, x, mu) for x in x_flat for mu in mu_list])
        scatt_resh = np.array(scatt).reshape(x_list.shape + (len(mu_list), 2))
        return scatt_resh[..., 0], scatt_resh[..., 1]  # S1, S2

    @staticmethod
    def _get_extinction_coeff(m: list, w: list, s: list) -> np.ndarray:
        # mieq = lambda mm, x, mu: MieQ(mm, x, mu, asCrossSection=True)
        with Pool() as p:
            scatt = p.starmap(MieQ, [(m, ww, ss, 1, False, True) for ww in w for ss in s])
        mie_q_res = np.array(scatt).reshape((len(w), len(s), 7))
        return mie_q_res


if __name__ == '__main__':
    spectrum = np.logspace(np.log10(300), np.log10(700), 10)  # white light wavelength in nm
    mie = MieScatteringModel.train(spectrum)

    mie.save('save_test')
    mie_loaded = MieScatteringModel.load('save_test.npz')
    os.remove('save_test.npz')

    wavelength_test = 500
    theta_test = np.linspace(0, np.pi, 100)
    SL, SR, SU, P = mie.get_scattering(wavelength_test, theta_test)
    mie_scatt = mie(wavelength_test, theta_test)

    # plot the Mueller matrix elements
    plot_mueller_matrix_elems(theta_test, mie_scatt[..., 0, 0], mie_scatt[..., 0, 1],
                              mie_scatt[..., 2, 2], mie_scatt[..., 2, 3])
    plot_intensity_polarization(theta_test, SL, SR, SU, P)

