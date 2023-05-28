import numpy as np


def get_emission_mueller_matrix():
    """
    Get the emission Mueller matrix
    :return: Mueller matrix
    """
    M = np.zeros((1, 4, 4), dtype=np.complex128)
    M[0, 0, 0] = 1  # unpolarized emission
    return M


def get_unpolarized_stokes_vector():
    """
    Get the Stokes vector for unpolarized emission
    :return: Stokes vector
    """
    S = np.zeros((1, 4, 1), dtype=np.float64)
    S[0, 0, 0] = 1  # unpolarized emission
    return S


def get_rotation_mueller_matrix(rotation_angle: np.ndarray) -> np.ndarray:
    """
    Get the Mueller matrix for rotation
    :param rotation_angle: rotation angle in rad
    :return: Mueller matrix
    """
    M = np.zeros((len(rotation_angle), 4, 4), dtype=np.float64)
    M[:, 0, 0] = 1
    M[:, 1, 1] = np.cos(2 * rotation_angle)
    M[:, 1, 2] = np.sin(2 * rotation_angle)
    M[:, 2, 1] = -np.sin(2 * rotation_angle)
    M[:, 2, 2] = np.cos(2 * rotation_angle)
    M[:, 3, 3] = 1
    return M
