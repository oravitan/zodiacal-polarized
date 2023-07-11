"""
This file contains the base class for the calibration of zodipol created images.
"""
import abc
import numpy as np
import astropy.units as u

from tqdm import tqdm
from scipy.ndimage import uniform_filter

from zodipol.zodipy_local.zodipy.zodipy import IQU_to_image
from zodipol.zodipol.zodipol import Zodipol
from zodipol.zodipol.observation import Observation
from zodipol.utils.argparser import ArgParser


class BaseCalibration:
    """
    This class is used to calibrate zodipol created images.
    The base class is the general calibration class. Other classes inherit from this class.
    """
    def __init__(self, zodipol: Zodipol, parser: ArgParser):
        """
        Initialize the calibration class.
        """
        self.obs = None
        self.p = None
        self.eta = None
        self.biref = None
        self.init = None
        self.zodipol = zodipol
        self.parser = parser
        self.initialize()

    def initialize(self, init: dict = None):
        """
        Initialize the parameters of the calibration.
        :param init: dictionary with initial values for the parameters
        """
        init = (init if init is not None else {})

        # initialize the parameters
        self.p = np.ones(self.parser["resolution"]).reshape(-1, 1).repeat(self.parser["n_polarization_ang"], axis=-1)
        self.eta = np.zeros(self.parser["resolution"]).reshape((-1))
        delta = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)
        alpha = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)

        # apply the init dictionary
        self.p = (init['p'] if 'p' in init else self.p)
        self.eta = (init['eta'] if 'eta' in init else self.eta)
        delta = (init['delta'] if 'delta' in init else delta)
        alpha = (init['alpha'] if 'alpha' in init else alpha)

        # create the birefringence matrix
        self.biref = self.zodipol.imager.get_birefringence_mueller_matrix(delta, alpha)[..., :3, :3]
        self.biref = (init['biref'] if 'biref' in init else self.biref)
        self.init = {'p': self.p, 'eta': self.eta, 'biref': self.biref}

    def forward_model(self, obs: Observation) -> u.Quantity:
        """
        Calculate the forward model of the calibration. This turns a stokes object into an image.
        :param obs: The observation object.
        :return: The estimated image.
        """
        p = self.p.reshape((-1, self.parser["n_polarization_ang"]))
        eta = self.eta.reshape((-1, 1)) + self.parser["polarization_angle"][None, :]
        biref_obs = self.zodipol.imager.apply_birefringence(obs, self.biref)
        img_model = IQU_to_image(biref_obs.I, biref_obs.Q, biref_obs.U, p, eta)
        return img_model

    def get_rmse(self, images_orig: u.Quantity) -> float:
        """
        Calculate the mean squared error between the forward model and the original images.
        :param images_orig: The original images.
        :return: The mean squared error.
        """
        img_model = np.stack([self.forward_model(o) for o in self.obs], axis=-1)
        rmse = np.nanmean(np.nansum((img_model - images_orig) ** 2, axis=0) ** 0.5) / np.prod(self.parser["resolution"])
        A_gamma = self.zodipol.imager.get_A_gamma(self.zodipol.frequency, self.zodipol.get_imager_response())
        rmse_electrons = (rmse / A_gamma).si.value.squeeze()
        return rmse_electrons

    def calibrate(self, images_orig, n_itr=5, disable=False, callback=None, init=None, **kwargs) -> tuple:
        """
        Calibrate the imager using the images.
        :param images_orig: The original images.
        :param n_itr: The number of iterations.
        :param disable: Disable the progress bar.
        :param callback: A callback function that is called after each iteration.
        :param init: A dictionary with initial values for the parameters.
        """
        self.initialize(init)
        itr_callback, itr_cost = [], []
        if self.obs is not None:
            itr_cost.append(self.get_rmse(images_orig))
            if callback is not None:
                itr_callback.append(callback(self))
        for _ in tqdm(range(n_itr), disable=disable):
            self._calibrate_itr(images_orig, **kwargs)
            itr_cost.append(self.get_rmse(images_orig))
            if callback is not None:
                itr_callback.append(callback(self))
        if callback is not None:
            return self.p, self.eta, self.biref, itr_cost, itr_callback
        return self.p, self.eta, self.biref, itr_cost

    @abc.abstractmethod
    def _calibrate_itr(self, images: u.Quantity, **kwargs) -> None:
        """
        Perform one iteration of the calibration.
        """
        ...

    def estimate_polarizance(self, images: u.Quantity, kernel_size=None, star_pixels=None, remove_stars=True, **kwargs) -> None:
        """
        Estimate the polarization of every pixel.
        :param images: The images.
        :param kernel_size: The size of the kernel used for the uniform filter.
        :param star_pixels: The pixels that belong to the star.
        """
        intensity = images.value

        mueller = self.biref
        stokes = np.stack([o.to_numpy(ndims=3) for o in self.obs], axis=-2).value
        stokes_tag = np.einsum('...ij,...jk->...ik', stokes, mueller)

        # WLS
        if star_pixels is None:
            star_pixels = np.stack([o.star_pixels for o in self.obs], axis=-1)
        if remove_stars:
            stokes_tag = np.einsum('ij...,ij->ij...', stokes_tag, ~star_pixels)
            intensity = np.einsum('i...j,ij->i...j', intensity, ~star_pixels)

        stokes_I, stokes_QU = stokes_tag[..., 0], stokes_tag[..., 1:]
        intensity_I = np.moveaxis(intensity, -2, -1) - 0.5 * stokes_I[..., None]
        intensity_I = intensity_I.reshape((intensity_I.shape[0], -1))

        S_P = 0.5 * np.concatenate((stokes_QU, -stokes_QU), axis=-1).reshape(intensity_I.shape)[..., None]

        pseudo_inv = np.linalg.pinv(S_P)
        p_est = np.einsum('...ij,...j->...i', pseudo_inv, intensity_I)

        p_est = self._pose_constraints_p(p_est)
        p_est = p_est.repeat(4, axis=-1)
        self.p = p_est

    @staticmethod
    def _pose_constraints_p(p):
        p = np.clip(p, 0, 1)
        return p

    def estimate_birefringence(self, images: u.Quantity, kernel_size: int = None, normalize_eigs: bool = False,
                               star_pixels=None, remove_stars=True, **kwargs) -> None:
        """
        Estimate the birefringence of every pixel.
        :param images: The images.
        :param kernel_size: The size of the kernel used for the uniform filter.
        :param normalize_eigs: Normalize the eigenvalues of the birefringence matrix.
        :param star_pixels: The pixels that belong to the star.
        """
        intensity = images.value.swapaxes(-1, -2)
        p = self.p  #[:, None]
        angles = self.eta[:, None] + self.parser["polarization_angle"]

        # preparation
        stokes = np.stack([o.to_numpy(ndims=3) for o in self.obs], axis=-1).value

        # WLS
        if star_pixels is None:
            star_pixels = np.stack([o.star_pixels for o in self.obs], axis=-1)
        if remove_stars:
            stokes = np.einsum('i...j,ij->i...j', stokes, ~star_pixels)
            intensity = np.einsum('ij...,ij->ij...', intensity, ~star_pixels)

        stokes_I, stokes_QU = stokes[:, 0, :], stokes[:, 1:, :]
        intensity_I = intensity - 0.5 * stokes_I[..., None]

        angles_matrix = 0.5 * np.stack((p * np.cos(2 * angles), p * np.sin(2 * angles)), axis=-1)

        g_k, f_k = stokes_QU[..., 0, :], stokes_QU[..., 1, :]

        F1 = np.stack((g_k, f_k, np.zeros_like(g_k)), axis=-1)
        F2 = np.stack((np.zeros_like(g_k), g_k, f_k), axis=-1)
        F_K1, F_K2, F_K3, F_K4 = angles_matrix[:, 0, 0, None, None] * F1, angles_matrix[:, 1, 1, None, None] * F2, \
            angles_matrix[:, 2, 0, None, None] * F1, angles_matrix[:, 3, 1, None, None] * F2
        F_K = np.stack((F_K1, F_K2, F_K3, F_K4), axis=-2).reshape((F_K1.shape[0], -1, 3))

        N_I = intensity_I.reshape((intensity_I.shape[0], -1))

        F_k_inv = np.linalg.pinv(F_K)
        biref_elems = np.einsum('...ij,...j->...i', F_k_inv, N_I)

        biref = np.stack((np.stack((biref_elems[..., 0], biref_elems[..., 1]), axis=-1),
                          np.stack((biref_elems[..., 1], biref_elems[..., 2]), axis=-1)), axis=-1)

        if normalize_eigs:
            biref = self._biref_normalize_eigs(biref)

        # smooth biref
        if kernel_size is not None:
            biref = self._biref_smooth_kernel(biref, kernel_size, resolution=self.parser["resolution"])

        biref = np.clip(biref, -1, 1)

        # set necessary values of biref
        biref_full = np.eye(3)[None, :].repeat(self.biref.shape[0], axis=0)
        biref_full[..., 1:, 1:] = biref
        self.biref = biref_full

    @staticmethod
    def _biref_normalize_eigs(biref_elems):
        """
        Normalize the eigenvalues of the birefringence matrix.
        :param biref_elems: The birefringence matrix.
        :return: The normalized birefringence matrix.
        """
        W, V = np.linalg.eig(biref_elems)
        biref_eigs = W / W.max(axis=1, keepdims=True)
        biref_res = np.einsum('...ij,...j,...kj->...ik', V, biref_eigs, V)

        biref_res = np.nan_to_num(biref_res, nan=0)
        return biref_res

    @staticmethod
    def _biref_smooth_kernel(biref: np.ndarray, kernel_size: int, resolution):
        """
        Smooth the birefringence matrix.
        :param biref: The birefringence matrix.
        :param kernel_size: The size of the kernel used for the uniform filter.
        :param resolution: The resolution of the image.
        """
        biref_resh = biref.reshape(resolution + [np.prod(biref.shape[1:])])
        biref_smooth = np.stack([uniform_filter(biref_resh[..., ii], size=kernel_size, mode='nearest') for ii in
                                 range(biref_resh.shape[-1])], axis=-1)
        biref = biref_smooth.reshape(biref.shape)
        return biref
