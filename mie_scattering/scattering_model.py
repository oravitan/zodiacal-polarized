import numpy as np

from multiprocessing import Pool
from tqdm import tqdm
from PyMieScatt import MieS1S2


def get_scattering_function(m: list, x_list: list, mu_list: list) -> (np.ndarray, np.ndarray):
    """
    Get the scattering function for a given refractive index, size parameter, and scattering angle,
    Using synced parallel processing to speed up the calculation.
    :param m: refractive index
    :param x_list: list of size parameter
    :param mu_list: list of cosine of scattering angle
    :return: scattering function (size param x cos scattering angle x refractive index x S1/S2)
    """
    with Pool() as p:
        scatt = p.starmap(MieS1S2, [(mm, x, mu) for x in x_list for mu in mu_list for mm in m])
    scatt_resh = np.array(scatt).reshape((len(x_list), len(mu_list), len(m), 2))
    return scatt_resh[..., 0], scatt_resh[..., 1]  # S1, S2


def calculate_S1S2(s, n_norm, spectrum, mu, refractive_ind: dict) -> list:
    """
    Calculate the scattering function for a given particle size distribution, spectrum, and scattering angle.
    :param s: particle size
    :param n_norm: normalized particle size distribution
    :param spectrum: spectrum
    :param mu: cosine of scattering angle
    :param refractive_ind: refractive index
    :return: scattering function (size param x cos scattering angle x refractive index x S1/S2)
    """
    refractive_ind_list = list(refractive_ind.keys())
    refractive_ind_prc = np.array(list(refractive_ind.values()))
    scat_functions = []
    for w in tqdm(spectrum):
        x = np.pi * s / w  # size parameter
        S1, S2 = get_scattering_function(refractive_ind_list, x, mu)  # scattering functions of graphite
        S1_avg = np.sum((S1 * refractive_ind_prc).sum(axis=-1) * n_norm[..., None], axis=0)
        S2_avg = np.sum((S2 * refractive_ind_prc).sum(axis=-1) * n_norm[..., None], axis=0)
        scat_functions.append({'S1': S1_avg, 'S2': S2_avg, 'w': w})
    return scat_functions

