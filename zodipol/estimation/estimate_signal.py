import numpy as np
import astropy.units as u


def estimate_IQU(intensity, polarizance, angles):
    """
    Calculate the Stokes parameters of the signal.
    :param intensity: The intensity of the signal.
    :param angles: The angles of the signal.
    :return: The Stokes parameters of the signal.
    """
    # create the angles matrix
    angles_matrix = 0.5 * np.stack((np.ones_like(angles), polarizance*np.cos(2*angles), polarizance*np.sin(2*angles)), axis=-2)
    pseudo_inverse = np.einsum('...ij,...kj->...ik', angles_matrix, angles_matrix)
    intensity_mult = np.einsum('...ij,...j->...i', angles_matrix, intensity)
    intensity_inv = np.linalg.solve(pseudo_inverse, intensity_mult)
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
    return np.sqrt(U ** 2 + Q ** 2) / (I + 1e-80 * I.unit)


def estimate_AoP(Q, U):
    """
    Calculate the angle of linear polarization.
    :param I: The intensity of the signal.
    :param U: The U component of the signal.
    :param Q: The Q component of the signal.
    :return: The degree of linear polarization.
    """
    return 0.5 * np.arctan(U / (Q + 1e-80 * Q.unit))
