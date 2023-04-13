import numpy as np

from skimage.filters import gaussian
from scipy.stats import norm
from skimage.transform import rotate

from zodipol.zodipy_local.zodipy.zodipy import IQU_to_image
from zodipol.estimation.estimate_signal import estimate_DoLP, estimate_AoP, estimate_IQU
from zodipol.mie_scattering.mueller_matrices import get_rotation_mueller_matrix


class Observation:
    def __init__(self, I, Q, U, theta=None, phi=None, roll=0):
        assert I.size == Q.size == U.size, 'I, Q and U must have the same length.'
        assert theta is None or theta.size == I.shape[0], 'Theta must have the same length as I, Q and U.'
        assert phi is None or phi.size == I.shape[0], 'Phi must have the same length as I, Q and U.'
        self.I = I
        self.Q = Q
        self.U = U
        self.theta = theta
        self.phi = phi
        self.roll = roll

    def __len__(self):
        return len(self.I)

    @classmethod
    def from_image(cls, image, polarizance, polarization_angle):
        polarization_angle = np.broadcast_to(polarization_angle, image.shape)
        I, Q, U = estimate_IQU(image, polarizance, polarization_angle)
        return cls(I, Q, U)

    def to_numpy(self, ndims=4):
        if ndims == 3:
            return np.stack([self.I, self.Q, self.U], axis=-1)
        if ndims == 4:
            V = np.zeros_like(self.I)
            return np.stack([self.I, self.Q, self.U, V], axis=-1)
        else:
            raise ValueError('Number of dimensions must be equal to 3 or 4')

    def mul(self, other: np.ndarray):
        if other.shape[-2] != 3 and other.shape[-1] != 3:
            raise ValueError(f'Cannot multiply observation with shape {(self.I.shape) + (3,)} with array with shape {other.shape}.')
        total_obs = np.stack([self.I, self.Q, self.U], axis=-1)
        total_obs_mul = np.einmul('...ij,...jk->...ik', other, total_obs)
        return Observation(total_obs_mul[..., 0], total_obs_mul[..., 1], total_obs_mul[..., 2])

    def copy(self):
        return Observation(self.I, self.Q, self.U, theta=self.theta, phi=self.phi, roll=self.roll)

    def change_roll(self, new_roll):
        rot_mat = get_rotation_mueller_matrix(np.array([new_roll - self.roll]))
        rotated_elem = np.einsum('...ij,...j->...i', rot_mat[0, :3, :3], self.to_numpy(ndims=3))
        I, Q, U = rotated_elem[..., 0], rotated_elem[..., 1], rotated_elem[..., 2]
        return Observation(I, Q, U, theta=self.theta, phi=self.phi, roll=new_roll)

    def get_binned_emission(self, polarization_angle: np.ndarray, polarizance: float):
        return IQU_to_image(self.I, self.U, self.Q, polarization_angle, polarizance)

    def get_dolp(self):
        return estimate_DoLP(self.I, self.Q, self.U)

    def get_aop(self):
        return estimate_AoP(self.Q, self.U)

    def add_direction_uncertainty(self, fov, resolution, direction_uncertainty):
        pixel_size = fov / resolution
        pixels_uncertainty = (direction_uncertainty / pixel_size).value
        pixels_uncertainty = np.concatenate((pixels_uncertainty, [1]))
        prev_shape = self.I.shape
        new_shape = resolution + list(self.I.shape[1:])
        I_g, Q_g, U_g = [gaussian(x.reshape(new_shape), pixels_uncertainty).reshape(prev_shape) for x in (self.I, self.Q, self.U)]
        I_g, Q_g, U_g = [x * self.I.unit for x in (I_g, Q_g, U_g)]
        return Observation(I_g, Q_g, U_g, theta=self.theta, phi=self.phi, roll=self.roll)

    def add_radial_blur(self, radial_blur, resolution):
        if radial_blur.value < 0:
            raise ValueError('Circular motion blur must be positive')
        elif radial_blur.value == 0:
            return self.copy()
        x = np.linspace(norm.ppf(0.01, scale=radial_blur), norm.ppf(0.99, scale=radial_blur), 10)
        norm_value = norm.pdf(x, scale=radial_blur)
        I = self._add_rotation(self.I, x, norm_value, resolution)
        Q = self._add_rotation(self.Q, x, norm_value, resolution)
        U = self._add_rotation(self.U, x, norm_value, resolution)
        return Observation(I, Q, U, theta=self.theta, phi=self.phi, roll=self.roll)

    def _add_rotation(self, I, x, norm_values, resolution):
        norm_values /= np.sum(norm_values)
        units = I.unit
        shape_old = I.shape
        shape_new = resolution + list(I.shape[1:])
        I_new = np.stack([rotate(I.reshape(shape_new), x_val, mode='edge') for x_val in x], axis=0)
        I_mean = np.nansum(I_new * np.expand_dims(norm_values, tuple(range(1, I_new.ndim))), axis=0)
        I = I_mean.reshape(shape_old)
        return I * units
