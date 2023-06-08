"""
This module contains the SelfCalibration class, which is used to self-calibrate
zodipol created images.
"""
import numpy as np
import astropy.units as u

from zodipol.utils.math import align_images, get_rotated_image
from zodipol.estimation.base_calibration import BaseCalibration
from zodipol.zodipol import Observation
from zodipol.utils.mueller_matrices import get_rotation_mueller_matrix


class SelfCalibration(BaseCalibration):
    """
    This class is used to self-calibrate zodipol created images.
    """
    def __init__(self, images_res_flat, rotation_list, zodipol, parser, theta=None, phi=None):
        super().__init__(zodipol, parser)

        self.rotation_list = rotation_list
        self.theta = theta
        self.phi = phi
        self.nobs = len(self.rotation_list)
        self.aligned_images = align_images(zodipol, parser, images_res_flat.value, rotation_list, fill_value=0)
        self.nan_mask = self.get_nan_mask(images_res_flat)
        self.initialize()

    def calibrate(self, images_orig, n_itr=5, disable=False, callback=None, init=None, **kwargs):
        """
        Calibrate the images.
        """
        self.initialize(init)
        self.obs = self.estimate_observations(images_orig)
        return super().calibrate(images_orig, n_itr=n_itr, disable=disable, callback=callback, init=init, **kwargs)

    def _calibrate_itr(self, images, **kwargs):
        self.obs = self.estimate_observations(images)
        self.estimate_polarizance(images, **kwargs)
        self.estimate_birefringence(images, **kwargs)

    def estimate_polarizance(self, images, **kwargs):
        super().estimate_polarizance(images)
        max_p = (kwargs['max_p'] if 'max_p' in kwargs else 1)
        self.p = self.p - np.quantile(self.p[~self.nan_mask], 0.95) + max_p
        self.p = np.clip(self.p, 0, 1)

    def estimate_birefringence(self, images, kernel_size: int = None, normalize_eigs: bool = False, **kwargs):
        super().estimate_birefringence(images, kernel_size=kernel_size, normalize_eigs=normalize_eigs)

    def get_nan_mask(self, images):
        """
        Get a mask of the images that are nan, because they're not contained within all input images.
        """
        nan_imag = align_images(self.zodipol, self.parser, np.ones((images.shape[0], len(self.rotation_list))), self.rotation_list, fill_value=np.nan)
        nan_ind = np.isnan(nan_imag).any(axis=-1).squeeze()
        return nan_ind

    def get_properties(self):
        """
        Get the properties of the calibration.
        """
        images = np.stack([self.forward_model(o) for o in self.obs], axis=-1)
        images[self.nan_mask, ...] = np.nan
        res = [images]
        for ii in [self.p, self.eta, self.biref]:
            ii_copy = ii.copy()
            ii_copy[self.nan_mask, ...] = np.nan
            res.append(ii_copy)
        return tuple(res)

    # Stokes vectors estimation
    def estimate_observations(self, images):
        """
        Estimate the observations Stokes vectors from the images.
        """
        rotation_list = self.rotation_list
        interp_images_res = self.aligned_images
        p = align_images(self.zodipol, self.parser, self.p.repeat(self.nobs, axis=-1), rotation_list, invert=True)[..., None, :].repeat(self.parser["n_polarization_ang"], axis=-2)
        eta = align_images(self.zodipol, self.parser, self.eta[:, None, None].repeat(self.nobs, axis=-1), rotation_list, invert=True) + self.parser["polarization_angle"][:, None]
        mueller = self.biref
        mueller_rot = mueller[..., None].repeat(self.nobs, axis=-1)
        mueller_rot = align_images(self.zodipol, self.parser, mueller_rot, rotation_list, invert=True)
        rotation_mat = get_rotation_mueller_matrix(np.deg2rad(rotation_list))[None, ...].repeat(images.shape[0], axis=0)
        I_est, Q_est, U_est = self.estimate_IQU(interp_images_res, rotation_mat, p, eta, mueller_rot)
        obs_est = Observation(I_est, Q_est, U_est, theta=self.theta, phi=self.phi, roll=0)
        obs_est_rot = [self.realign_observation(obs_est, roll) for roll in rotation_list]
        return obs_est_rot

    @staticmethod
    def estimate_IQU(intensity, rotation_mat, polarizance, angles, mueller):
        """
        Calculate the Stokes parameters of the signal.
        :param intensity: The intensity of the signal.
        :param angles: The angles of the signal.
        :return: The Stokes parameters of the signal.
        """
        angles_matrix = 0.5 * np.stack((np.ones_like(angles), polarizance*np.cos(2*angles), polarizance*np.sin(2*angles)), axis=-3)
        forward_mat = np.einsum('...ijk,...iwk,...kws->...sjk', angles_matrix, mueller[:, :3, :3, :], rotation_mat[..., :3, :3])

        forward_resh = forward_mat.reshape(forward_mat.shape[:2] + (np.prod(forward_mat.shape[2:]),))
        intensity_resh = intensity.reshape((intensity.shape[0], np.prod(intensity.shape[1:])))

        pseudo_inverse = np.linalg.pinv(forward_resh)
        stokes_inv = np.einsum('...ij,...i->...j', pseudo_inverse, intensity_resh)
        stokes_inv = stokes_inv << u.Unit('W / m^2 sr')  # set Stokes vector units
        I, Q, U = stokes_inv[..., 0], stokes_inv[..., 1], stokes_inv[..., 2]
        return I, Q, U

    def realign_observation(self, obs, roll):
        """
        Realign the observation from the reference frame to their original frame.
        """
        stokes_interp = get_rotated_image(self.zodipol, self.parser, obs.to_numpy(ndims=3), roll)
        obs_new = Observation(stokes_interp[..., 0], stokes_interp[..., 1], stokes_interp[..., 2], theta=obs.theta,
                              phi=obs.phi, roll=0)
        return obs_new.change_roll(np.deg2rad(roll))

    