import numpy as np

from zodipol.camera.camera_constants import exposure_time, std_read, beta_t, full_well, n_bits


def complete_noise_model(n_electrons):
    """
    Apply all the noise models to the number of electrons.
    :param n_electrons: number of electrons
    :return: number of electrons with all the noise models
    """
    n_electrons_poiss = camera_poisson_noise(n_electrons)
    n_electrons_read = camera_readout_noise(n_electrons_poiss)
    n_electrons_dark = dark_current_noise(n_electrons_read)
    n_electrons_quant = quantization_noise(n_electrons_dark)
    n_electron_pos = np.where(n_electrons_quant > 0, n_electrons_quant, 0)
    return n_electron_pos


def camera_poisson_noise(n_electrons):
    """
    Add poisson noise to the number of electrons.
    :param n_electrons: number of electrons
    :return: number of electrons with poisson noise
    """
    camera_mask = np.isnan(n_electrons)
    noised_n_electrons = np.random.poisson(np.nan_to_num(n_electrons, nan=0))
    noised_n_electrons = noised_n_electrons.astype(np.float64)
    noised_n_electrons[camera_mask] = np.nan
    return noised_n_electrons


def camera_readout_noise(n_electrons):
    """
    Add readout noise to the number of electrons.
    :param n_electrons: number of electrons
    :return: number of electrons with readout noise
    """
    image_size = n_electrons.shape
    readout_noise = np.floor(np.random.normal(0, std_read, image_size))
    return n_electrons + readout_noise


def dark_current_noise(n_electrons):
    """
    Add dark current noise to the number of electrons.
    :param n_electrons: number of electrons
    :return: number of electrons with dark current noise
    """
    dark_current = - np.floor(np.random.normal(beta_t * exposure_time, np.sqrt(beta_t * exposure_time), n_electrons.shape))
    return n_electrons + dark_current


def quantization_noise(n_electrons):
    """
    Add full well noise to the number of electrons.
    :param n_electrons: number of electrons
    :return: number of electrons with full well noise
    """
    full_well_factor = 2 ** n_bits / full_well
    n_electrons_full_well = 1 / full_well_factor * np.floor(full_well_factor * n_electrons)
    return n_electrons_full_well
