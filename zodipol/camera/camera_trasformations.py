import numpy as np
import astropy.units as u
from astropy.constants import h

from zodipol.camera.camera_constants import exposure_time, pixel_area, lens_diameter, lens_focal_length, optical_loss, quantum_efficiency


def intensity_to_number_of_electrons(intensity, frequency, frequency_weight):
    """
    Calculate the number of electrons from the intensity.
    :param intensity: camera received intensity pixels
    :param frequency: frequency range
    :param frequency_weight: frequency weights
    :return: number of electrons
    """
    intensity = intensity.to('J / m^2 sr')  # transform to W / m^2 Hz sr
    focal_param = np.pi * (lens_diameter / 2 / lens_focal_length) ** 2 * u.sr  # Focal parameter
    energy = intensity * exposure_time * pixel_area * focal_param * optical_loss  # Energy
    n_electrons_per_freq = energy[..., None] / (h * frequency[None, None, :]) * quantum_efficiency  # Number of electrons per frequency

    n_electrons = np.trapz(n_electrons_per_freq * frequency_weight[None, None, ...], frequency)  # Number of electrons integral
    n_electrons = n_electrons.si  # Number of electrons in SI units
    return n_electrons


def number_of_electrons_to_intensity(n_electrons, frequency, frequency_weight):
    """
    Calculate the intensity from the number of electrons.
    :param n_electrons: number of electrons
    :param frequency: frequency range
    :param frequency_weight: frequency weights
    :return: camera received intensity pixels
    """
    focal_param = np.pi * (lens_diameter / 2 / lens_focal_length) ** 2 * u.sr  # Focal parameter
    gamma = optical_loss * focal_param * quantum_efficiency * pixel_area * (1 / (h * frequency[None, None, :]))
    toa_factor = 1 * u.s  # Top of atmosphere factor
    A_gamma = 1 / np.trapz(toa_factor * gamma * frequency_weight[None, None, ...], frequency) / exposure_time

    intensity = n_electrons * A_gamma  # Intensity
    return intensity.to('W / m^2 sr')
