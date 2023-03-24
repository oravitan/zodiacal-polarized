import numpy as np

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
    def from_image(cls, image, polarization_angle):
        I, Q, U = estimate_IQU(image, polarization_angle)
        return cls(I, Q, U)

    def to_numpy(self, ndims=4):
        if ndims == 3:
            return np.stack([self.I, self.Q, self.U], axis=-1)
        if ndims == 4:
            V = np.sqrt(self.I**2 - self.Q**2 - self.U**2)
            return np.stack([self.I, self.Q, self.U, V], axis=-1)
        else:
            raise ValueError('Number of dimensions must be equal to 3 or 4')

    def mul(self, other: np.ndarray):
        if other.shape[-2] != 3 and other.shape[-1] != 3:
            raise ValueError(f'Cannot multiply observation with shape {(self.I.shape) + (3,)} with array with shape {other.shape}.')
        total_obs = np.stack([self.I, self.Q, self.U], axis=-1)
        total_obs_mul = np.einmul('...ij,...jk->...ik', other, total_obs)
        return Observation(total_obs_mul[..., 0], total_obs_mul[..., 1], total_obs_mul[..., 2])

    def change_roll(self, new_roll):
        rot_mat = get_rotation_mueller_matrix(np.array([new_roll - self.roll]))
        rotated_elem = np.einsum('...ij,...j->...i', rot_mat, self.to_numpy(ndims=4))
        I, Q, U, _ = rotated_elem[..., 0], rotated_elem[..., 1], rotated_elem[..., 2], rotated_elem[..., 3]
        return Observation(I, Q, U, theta=self.theta, phi=self.phi, roll=new_roll)

    def get_binned_emission(self, polarization_angle: np.ndarray, polarizance: float):
        return IQU_to_image(self.I, self.U, self.Q, polarization_angle, polarizance)

    def get_dolp(self):
        return estimate_DoLP(self.I, self.Q, self.U)

    def get_aop(self):
        return estimate_AoP(self.Q, self.U)
