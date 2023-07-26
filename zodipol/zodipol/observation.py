"""
This module contains the Observation class, which is used to store the Stokes parameters of a single observation.
"""
import numpy as np

from skimage.filters import gaussian
from scipy.stats import norm
from skimage.transform import rotate
from skimage.morphology import dilation, disk

from zodipol.zodipy_local.zodipy.zodipy import IQU_to_image
from zodipol.estimation.estimate_signal import estimate_DoLP, estimate_AoP, estimate_IQU
from zodipol.utils.mueller_matrices import get_rotation_mueller_matrix


class Observation:
    def __init__(self, I, Q, U, theta=None, phi=None, roll=0, star_pixels=None):
        """
        :param I: Stokes I
        :param Q: Stokes Q
        :param U: Stokes U
        :param theta: theta angle of the observation
        :param phi: phi angle of the observation
        :param roll: roll angle of the observation
        :param star_pixels: binary map of star pixels
        """
        assert I.size == Q.size == U.size, 'I, Q and U must have the same length.'
        assert theta is None or theta.size == I.shape[0], 'Theta must have the same length as I, Q and U.'
        assert phi is None or phi.size == I.shape[0], 'Phi must have the same length as I, Q and U.'
        self.I = I
        self.Q = Q
        self.U = U
        self.theta = theta
        self.phi = phi
        self.roll = roll
        self.star_pixels = star_pixels  # binary map of star pixels

    def __len__(self):
        """
        :return: number of pixels in the observation
        """
        return len(self.I)

    @classmethod
    def from_image(cls, images, polarizance, polarization_angle, **kwargs):
        """
        Create an Observation object by Stokes estimation from images and its polarizance and polarization angle.
        :param images: images to estimate Stokes parameters from
        :param polarizance: polarizance of the imager
        :param polarization_angle: polarization angle of the imager
        :param kwargs: additional keyword arguments
        :return: Observation object
        """
        polarization_angle = np.broadcast_to(polarization_angle, images.shape)
        I, Q, U = estimate_IQU(images, polarizance, polarization_angle)
        return cls(I, Q, U, **kwargs)

    def to_numpy(self, ndims=4):
        """
        :param ndims: number of dimensions of the output array
        :return: numpy array of shape (..., 3) or (..., 4) containing the Stokes parameters
        """
        if ndims == 3:
            return np.stack([self.I, self.Q, self.U], axis=-1)
        if ndims == 4:
            V = np.zeros_like(self.I)
            return np.stack([self.I, self.Q, self.U, V], axis=-1)
        else:
            raise ValueError('Number of dimensions must be equal to 3 or 4')

    def mul(self, other: np.ndarray):
        """
        Multiply the observation with a 3x3 array.
        :param other: array to multiply the observation with
        :return: new Observation object
        """
        if other.shape[-2] != 3 and other.shape[-1] != 3:
            raise ValueError(f'Cannot multiply observation with shape {(self.I.shape) + (3,)} with array with shape {other.shape}.')
        total_obs = np.stack([self.I, self.Q, self.U], axis=-1)
        total_obs_mul = np.einmul('...ij,...jk->...ik', other, total_obs)
        return Observation(total_obs_mul[..., 0], total_obs_mul[..., 1], total_obs_mul[..., 2])

    def copy(self):
        """
        :return: copy of the Observation object
        """
        return Observation(self.I, self.Q, self.U, theta=self.theta, phi=self.phi, roll=self.roll, star_pixels=self.star_pixels)

    def change_roll(self, new_roll):
        """
        Change the roll angle of the observation.
        :param new_roll: new roll angle
        :return: new Observation object
        """
        rot_mat = get_rotation_mueller_matrix(np.array([new_roll - self.roll]))
        rotated_elem = np.einsum('...ij,...j->...i', rot_mat[0, :3, :3], self.to_numpy(ndims=3))
        I, Q, U = rotated_elem[..., 0], rotated_elem[..., 1], rotated_elem[..., 2]
        return Observation(I, Q, U, theta=self.theta, phi=self.phi, roll=new_roll, star_pixels=self.star_pixels)

    def get_binned_emission(self, polarization_angle: np.ndarray, polarizance: float):
        """
        Get the binned emission of the observation.
        :param polarization_angle: polarization angle of the imager
        :param polarizance: polarizance of the imager
        :return: binned emission image
        """
        return IQU_to_image(self.I, self.U, self.Q, polarization_angle, polarizance)

    def get_dolp(self):
        """
        Estimate the degree of linear polarization of the observation.
        :return: degree of linear polarization
        """
        return estimate_DoLP(self.I, self.Q, self.U)

    def get_aop(self):
        """
        Estimate the angle of polarization of the observation.
        :return: angle of polarization
        """
        return estimate_AoP(self.Q, self.U)

    def add_direction_uncertainty(self, fov, resolution, direction_uncertainty):
        """
        Add direction uncertainty to the observation.
        :param fov: field of view of the observation
        :param resolution: resolution of the observation
        :param direction_uncertainty: amount of direction uncertainty of the observation (degrees)
        :return: new Observation object
        """
        if direction_uncertainty.value < 0:
            raise ValueError('Circular motion blur must be positive')
        elif direction_uncertainty.value == 0:
            return self.copy()
        pixel_size = fov / min(resolution)
        pixels_uncertainty = (direction_uncertainty / pixel_size).value
        pixels_uncertainty = np.array((pixels_uncertainty, pixels_uncertainty, 1))
        prev_shape = self.I.shape
        new_shape = resolution + list(self.I.shape[1:])
        I_g, Q_g, U_g = [gaussian(x.reshape(new_shape), pixels_uncertainty).reshape(prev_shape) for x in (self.I, self.Q, self.U)]
        I_g, Q_g, U_g = [x * self.I.unit for x in (I_g, Q_g, U_g)]
        return Observation(I_g, Q_g, U_g, theta=self.theta, phi=self.phi, roll=self.roll, star_pixels=self.star_pixels)

    def add_radial_blur(self, radial_blur, resolution):
        """
        Add radial blur to the observation.
        :param radial_blur: amount of radial blur of the observation (degrees)
        :param resolution: resolution of the observation
        :return: new Observation object
        """
        if radial_blur.value < 0:
            raise ValueError('Circular motion blur must be positive')
        elif radial_blur.value == 0:
            return self.copy()
        min_ppf, max_ppf = norm.ppf(0.01, scale=radial_blur), norm.ppf(0.99, scale=radial_blur)
        n_angs = np.max((np.ceil(max_ppf - min_ppf).astype(int), 11))
        x = np.linspace(min_ppf, max_ppf, n_angs, endpoint=True)
        norm_value = norm.pdf(x, scale=radial_blur)
        norm_value /= norm_value.sum()

        obs_rot_list = [self.rotate_observation(xx, resolution) for xx in x]
        obs_res = np.stack([o.to_numpy(ndims=3) for o in obs_rot_list], axis=0)
        obs_res_w = np.nansum(obs_res * np.expand_dims(norm_value, tuple(range(1, obs_res.ndim))), axis=0) * self.I.unit
        return Observation(obs_res_w[..., 0], obs_res_w[..., 1], obs_res_w[..., 2], theta=self.theta, phi=self.phi, roll=self.roll, star_pixels=self.star_pixels)

    def rotate_observation(self, rotation_angle, resolution):
        """
        Rotate the observation around the center of the image.
        :param rotation_angle: rotation angle
        :param resolution: resolution of the observation
        :return: new Observation object
        """
        obs_new = self.change_roll(self.roll - np.deg2rad(rotation_angle))
        obs_new_arr = obs_new.to_numpy(ndims=3)
        shape_old = obs_new_arr.shape
        shape_new = resolution + [np.prod(obs_new_arr.shape[1:])]
        obs_rot_arr = rotate(obs_new_arr.reshape(shape_new), rotation_angle, mode='edge').reshape(shape_old)
        I_new, Q_new, U_new = obs_rot_arr[..., 0], obs_rot_arr[..., 1], obs_rot_arr[..., 2]
        return Observation(I_new, Q_new, U_new, theta=self.theta, phi=self.phi, roll=self.roll - rotation_angle, star_pixels=self.star_pixels)

    def dilate_star_pixels(self, n_pixels, resolution):
        """
        Dilate the star pixels of the existing observation.
        :param n_pixels: number of pixels to dilate
        :param resolution: resolution of the observation
        """
        self_copy = self.copy()
        px = self_copy.star_pixels.reshape(resolution)
        px_dilate = dilation(px, disk(n_pixels))
        self_copy.star_pixels = px_dilate.reshape(self_copy.star_pixels.shape)
        return self_copy
