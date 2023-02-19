import numpy as np


def estimate_IQU(intensity, angles):
    """
    Calculate the Stokes parameters of the signal.
    :param intensity: The intensity of the signal.
    :param angles: The angles of the signal.
    :return: The Stokes parameters of the signal.
    """
    # create the angles matrix
    angles_matrix = np.stack((np.ones_like(angles), np.cos(2*angles), np.sin(2*angles)), axis=1)
    pseudo_inverse = angles_matrix.T @ angles_matrix
    intensity_mult = np.einsum('ij,...j->...i', angles_matrix.T, intensity)
    intensity_inv = np.einsum('ij,...j->...i', pseudo_inverse, intensity_mult)
    I, Q, U = intensity_inv[..., 0], intensity_inv[..., 1], intensity_inv[..., 2]
    return I, Q, U


def estimate_DoLP(I, U, Q):
    """
    Calculate the degree of linear polarization.
    :param I: The intensity of the signal.
    :param U: The U component of the signal.
    :param Q: The Q component of the signal.
    :return: The degree of linear polarization.
    """
    return np.sqrt(U ** 2 + Q ** 2) / I


def estimate_AoP(Q, U):
    """
    Calculate the angle of linear polarization.
    :param I: The intensity of the signal.
    :param U: The U component of the signal.
    :param Q: The Q component of the signal.
    :return: The degree of linear polarization.
    """
    return 0.5 * np.arctan2(U, Q)
