import abc
import numpy as np
import astropy.units as u

from tqdm import tqdm
from scipy.signal import convolve2d

from zodipol.zodipy_local.zodipy.zodipy import IQU_to_image
from zodipol.zodipol import Zodipol, Observation
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
        self.zodipol = zodipol
        self.parser = parser
        self.initialize()

    def initialize(self, init: dict = None):
        """
        Initialize the parameters of the calibration.
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

    def forward_model(self, obs: Observation) -> u.Quantity:
        """
        Calculate the forward model of the calibration.
        This turns a stokes object into an image.
        """
        p = self.p.reshape((-1, self.parser["n_polarization_ang"]))
        eta = self.eta.reshape((-1, 1)) + self.parser["polarization_angle"][None, :]
        biref_obs = self.zodipol.imager.apply_birefringence(obs, self.biref)
        img_model = IQU_to_image(biref_obs.I, biref_obs.Q, biref_obs.U, p, eta)
        return img_model

    def get_mse(self, images_orig: u.Quantity) -> float:
        """
        Calculate the mean squared error between the forward model and the original images.
        """
        img_model = np.stack([self.forward_model(o) for o in self.obs], axis=-1)
        mse = np.nanmean((img_model - images_orig) ** 2)
        A_gamma = self.zodipol.imager.get_A_gamma(self.zodipol.frequency, self.zodipol.get_imager_response())
        mse_electrons = (np.sqrt(mse) / A_gamma).si.value.squeeze() ** 2
        return mse_electrons

    def calibrate(self, images_orig, n_itr=5, disable=False, callback=None, init=None) -> tuple:
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
            itr_cost.append(self.get_mse(images_orig))
            if callback is not None:
                itr_callback.append(callback(self))
        for _ in tqdm(range(n_itr), disable=disable):
            self._calibrate_itr(images_orig)
            itr_cost.append(self.get_mse(images_orig))
            if callback is not None:
                itr_callback.append(callback(self))
        if callback is not None:
            return self.p, self.eta, self.biref, itr_cost, itr_callback
        return self.p, self.eta, self.biref, itr_cost

    @abc.abstractmethod
    def _calibrate_itr(self, images: u.Quantity) -> None:
        """
        Perform one iteration of the calibration.
        """
        pass

    def estimate_polarizance(self, images: u.Quantity) -> None:
        """
        Estimate the polarization of every pixel.
        """
        intensity = images.value

        mueller = self.biref
        stokes = np.stack([o.to_numpy(ndims=3) for o in self.obs], axis=-2).value
        stokes_tag = np.einsum('...ij,...jk->...ik', stokes, mueller)

        stokes_I, stokes_QU = stokes_tag[..., 0], stokes_tag[..., 1:]
        # V0 = 0.5 * np.concatenate((np.eye(2, 2), -np.eye(2, 2)))
        intensity_I = np.moveaxis(intensity, -2, -1) - 0.5 * stokes_I[..., None]
        intensity_I = intensity_I.reshape((intensity_I.shape[0], -1))

        # F_P = np.einsum('ijk,...wk->...ijw', stokes_QU.repeat(4, axis=1), V0)
        stokes_eye = np.einsum('...i,ij->...ij', stokes_QU, np.eye(2))
        S_P = np.kron(np.diag((1, -1)), stokes_eye)
        S_P = 0.5 * S_P.reshape(intensity_I.shape + (4,))

        pseudo_inv = np.linalg.pinv(S_P)
        p_est = np.einsum('...ij,...j->...i', pseudo_inv, intensity_I)

        p_est = np.clip(p_est, 0, 1)
        self.p = p_est

    def estimate_birefringence(self, images: u.Quantity, kernel_size: int = None, normalize_eigs: bool = False) -> None:
        """
        Estimate the birefringence of every pixel.
        """
        intensity = images.value.swapaxes(-1, -2)
        p = self.p  #[:, None]
        angles = self.eta[:, None] + self.parser["polarization_angle"]

        # preparation
        stokes = np.stack([o.to_numpy(ndims=3) for o in self.obs], axis=-1).value
        stokes_I, stokes_QU = stokes[:, 0, :], stokes[:, 1:, :]
        intensity_I = intensity - 0.5 * stokes_I[..., None]

        angles_matrix = 0.5 * np.stack((p * np.cos(2 * angles), p * np.sin(2 * angles)), axis=-1)

        stokes_pseudo_inv = np.linalg.pinv(stokes_QU.swapaxes(-1,-2))
        angles_pseudo_inv = np.linalg.pinv(angles_matrix.swapaxes(-1,-2))

        biref = np.einsum('...ij,...jk,...kw->...iw', stokes_pseudo_inv, intensity_I, angles_pseudo_inv)

        # assert np.sqrt(np.median((biref[:, 0, 0] - 1)**2)) < 0.05, 'biref[0, 0] is not 1'

        if normalize_eigs:
            W, V = np.linalg.eig(biref)
            eig_normalization = np.max((np.ones(biref.shape[:1]), np.max(W.real, axis=-1)), axis=0)[:, None, None]
            biref = biref / eig_normalization

        # smooth biref
        if kernel_size is not None:
            kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
            biref_resh = biref.reshape(self.parser["resolution"] + [9])
            biref_smooth = np.stack([convolve2d(biref_resh[..., ii], kernel, mode='same', boundary='symm') for ii in
                                     range(biref_resh.shape[-1])], axis=-1)
            biref = biref_smooth.reshape(biref.shape)

        # set necessary values of biref
        # biref_fixed = np.clip(biref, -1, 1).reshape(biref.shape)
        biref_full = np.eye(3)[None, :].repeat(self.biref.shape[0], axis=0)
        biref_full[..., 1:, 1:] = biref
        self.biref = biref_full
