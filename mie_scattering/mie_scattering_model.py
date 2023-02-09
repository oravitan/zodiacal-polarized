"""
Use the output of the Mie code to calculate the scattering Mueller matrix
"""
import numpy as np
import os
from mie_scattering.plotting import plot_mueller_matrix_elems


def get_mie_scattering():
    """
    Get the Mie scattering Mueller matrix elements
    :return: scattering angle in rad, S11, S12, S33, S34
    """
    file_location = os.path.join(os.path.dirname(__file__), 'outputs/mie_scattering.csv')
    theta, S11, S12, S33, S34 = np.loadtxt(file_location, delimiter=',', skiprows=1, unpack=True)
    return theta, S11, S12, S33, S34


def get_mie_scattering_mueller_elem(theta: np.ndarray):
    """
    Get the mie scattering mueller matrix elements for some theta
    :param theta: scattering angle in rad
    :return: S11, S12, S33, S34
    """
    theta_, S11_, S12_, S33_, S34_ = get_mie_scattering()
    S11 = np.interp(theta, theta_, S11_)
    S12 = np.interp(theta, theta_, S12_)
    S33 = np.interp(theta, theta_, S33_)
    S34 = np.interp(theta, theta_, S34_)
    return S11, S12, S33, S34


def get_mueller_matrix_from_elem(S11: np.ndarray, S12: np.ndarray, S33: np.ndarray, S34: np.ndarray):
    """
    Get the Mueller matrix
    :param S11: S11 element
    :param S12: S12 element
    :param S33: S33 element
    :param S34: S34 element
    :return: Mueller matrix
    """
    assert len(S11) == len(S12) == len(S33) == len(S34), 'S11, S12, S33, S34 must have the same length'
    M = np.zeros((4, 4, len(S11)), dtype=np.float64)
    M[0, 0, :] = S11
    M[0, 1, :] = S12
    M[1, 0, :] = S12
    M[1, 1, :] = S11

    M[2, 2, :] = S33
    M[2, 3, :] = -S34
    M[3, 2, :] = S34
    M[3, 3, :] = S33
    return M


def get_mie_scattering_mueller_matrix(theta: np.ndarray) -> np.ndarray:
    """
    Get the Mueller matrix for some theta in rad
    :param theta: scattering angle in rad
    :return: Mueller matrix
    """
    S11, S12, S33, S34 = get_mie_scattering_mueller_elem(theta)
    return get_mueller_matrix_from_elem(S11, S12, S33, S34)


def get_emission_mueller_matrix():
    """
    Get the emission Mueller matrix
    :return: Mueller matrix
    """
    M = np.zeros((4, 4, 1), dtype=np.complex128)
    M[0, 0, 0] = 1  # unpolarized emission
    return M


def get_unpolarized_stokes_vector():
    """
    Get the Stokes vector for unpolarized emission
    :return: Stokes vector
    """
    S = np.zeros((4, 1, 1), dtype=np.float64)
    S[0, 0] = 1  # unpolarized emission
    return S


def get_rotation_mueller_matrix(rotation_angle: np.ndarray) -> np.ndarray:
    """
    Get the Mueller matrix for rotation
    :param rotation_angle: rotation angle in rad
    :return: Mueller matrix
    """
    M = np.zeros((4, 4, len(rotation_angle)), dtype=np.float64)
    M[0, 0, :] = 1
    M[1, 1, :] = np.cos(2 * rotation_angle)
    M[1, 2, :] = np.sin(2 * rotation_angle)
    M[2, 1, :] = -np.sin(2 * rotation_angle)
    M[2, 2, :] = np.cos(2 * rotation_angle)
    M[3, 3, :] = 1
    return M


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    theta, S11, S12, S33, S34 = get_mie_scattering()

    # plot the Mueller matrix elements
    plot_mueller_matrix_elems(theta, S11, S12, S33, S34)

    # plot the Mueller matrix
    S11, S12, S33, S34 = get_mie_scattering_mueller_elem(np.array([0.1, 0.2, 0.3]))
    M = get_mueller_matrix_from_elem(S11, S12, S33, S34)
    print(M[..., 0])
    print(M[..., 1])
    print(M[..., 2])
