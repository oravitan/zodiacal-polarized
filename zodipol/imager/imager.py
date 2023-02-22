import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import h

from zodipol.utils.paths import IMAGER_RESPONSE_FILE_RED, IMAGER_RESPONSE_FILE_GREEN, IMAGER_RESPONSE_FILE_BLUE


IMAGER_RESPONSE_RED = pd.read_csv(IMAGER_RESPONSE_FILE_RED, index_col=0)
IMAGER_RESPONSE_GREEN = pd.read_csv(IMAGER_RESPONSE_FILE_GREEN, index_col=0)
IMAGER_RESPONSE_BLUE = pd.read_csv(IMAGER_RESPONSE_FILE_BLUE, index_col=0)


class Imager:
    def __init__(self,
                 exposure_time=10 * u.s,
                 pixel_area=(3.45 * u.um) ** 2,
                 lens_diameter=20.0 * u.mm,
                 lens_focal_length=86.2 * u.mm,
                 optical_loss=0.96,
                 quantum_efficiency=30,
                 std_read=2.31,
                 beta_t=3.51 * (u.s ** -1),
                 full_well=10500,
                 n_bits=10,
                 ):
        self.exposure_time = exposure_time
        self.pixel_area = pixel_area
        self.lens_diameter = lens_diameter
        self.lens_focal_length = lens_focal_length
        self.optical_loss = optical_loss
        self.quantum_efficiency = quantum_efficiency
        self.std_read = std_read
        self.beta_t = beta_t
        self.full_well = full_well
        self.n_bits = n_bits

    # ------------------------------------------------
    # ----------------- Response ---------------------
    # ------------------------------------------------
    def get_camera_response(self, wavelength: float, channel: str):
        """
        Get the imager response for a given wavelength and channel
        :param wavelength: wavelength in nm
        :param channel: channel (red, green, blue)
        :return: imager quantum efficiency response
        """
        if channel == 'red':
            return np.interp(wavelength, IMAGER_RESPONSE_RED.index, IMAGER_RESPONSE_RED['<Efficiency>'])
        elif channel == 'green':
            return np.interp(wavelength, IMAGER_RESPONSE_GREEN.index, IMAGER_RESPONSE_GREEN['<Efficiency>'])
        elif channel == 'blue':
            return np.interp(wavelength, IMAGER_RESPONSE_BLUE.index, IMAGER_RESPONSE_BLUE['<Efficiency>'])
        else:
            raise ValueError('Channel must be either red, green or blue')

    def get_wavelength_range(self, channel: str):
        if channel == 'red':
            return IMAGER_RESPONSE_RED.index
        elif channel == 'green':
            return IMAGER_RESPONSE_GREEN.index
        elif channel == 'blue':
            return IMAGER_RESPONSE_BLUE.index
        else:
            raise ValueError('Channel must be either red, green or blue')

    # ------------------------------------------------
    # ------------- Transformations ------------------
    # ------------------------------------------------
    def intensity_to_number_of_electrons(self, intensity, frequency=None, wavelength=None, weights=None):
        """
        Calculate the number of electrons from the intensity.
        :param intensity: imager received intensity pixels
        :param frequency: frequency range
        :param frequency_weight: frequency weights
        :return: number of electrons
        """
        assert frequency is not None or wavelength is not None, 'Either frequency or wavelength range must be provided'
        energy = self._get_energy(intensity)  # Energy

        # Calculate the number of electrons per pixel
        if frequency is not None:
            freq_gradient = np.gradient(frequency)
            n_electrons = np.einsum('ij...,j->i...', energy, freq_gradient / (h * frequency) * weights)  # Number of electrons per frequency
        else:
            frequency = wavelength.to(u.THz, equivalencies=u.spectral())
            wavelength_gradient = -np.gradient(wavelength)  # wavelength is sorted by frequency, so it inverse
            n_electrons = np.einsum('ij...,j->i...', energy, wavelength_gradient / (h * frequency) * weights)

        # n_electrons = np.sum(n_electrons_per_freq, axis=1)  # Number of electrons integral
        n_electrons = n_electrons.si  # Number of electrons in SI units
        return n_electrons

    def _get_energy(self, intensity):
        """
        Calculate the energy from the intensity.
        :param intensity: imager received intensity pixels
        :return: energy
        """
        focal_param = np.pi * (self.lens_diameter / 2 / self.lens_focal_length) ** 2 * u.sr  # Focal parameter
        energy_factor = self.exposure_time * self.pixel_area * focal_param * self.optical_loss
        return intensity * energy_factor  # Energy

    def number_of_electrons_to_intensity(self, n_electrons, frequency, frequency_weight):
        """
        Calculate the intensity from the number of electrons.
        :param n_electrons: number of electrons
        :param frequency: frequency range
        :param frequency_weight: frequency weights
        :return: imager received intensity pixels
        """
        focal_param = np.pi * (self.lens_diameter / 2 / self.lens_focal_length) ** 2 * u.sr  # Focal parameter
        gamma = self.optical_loss * focal_param * self.quantum_efficiency * self.pixel_area * (1 / (h * frequency[None, None, :]))
        toa_factor = 1 * u.s  # Top of atmosphere factor
        A_gamma = 1 / np.trapz(toa_factor * gamma * frequency_weight, frequency) / self.exposure_time

        intensity = n_electrons * A_gamma  # Intensity
        return intensity.to('W / m^2 sr')

    # ------------------------------------------------
    # ----------------- Noise models -----------------
    # ------------------------------------------------
    def imager_noise_model(self, n_electrons, poisson_noise=True, readout_noise=True, dark_current_noise=True,
                           quantization_noise=True):
        """
        Apply all the noise models to the number of electrons.
        :param n_electrons: number of electrons
        :return: number of electrons with all the noise models
        """
        if poisson_noise:
            n_electrons = self._camera_poisson_noise(n_electrons)
        if readout_noise:
            n_electrons = self._camera_readout_noise(n_electrons)
        if dark_current_noise:
            n_electrons = self._dark_current_noise(n_electrons)
        if quantization_noise:
            n_electrons = self._quantization_noise(n_electrons)
        n_electrons = self.camera_post_process(n_electrons)
        return n_electrons

    def camera_post_process(self, n_electrons):
        return np.clip(n_electrons, a_min=0, a_max=self.full_well)

    def camera_dark_current_estimation(self):
        return self.beta_t * self.exposure_time

    def _camera_poisson_noise(self, n_electrons):
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

    def _camera_readout_noise(self, n_electrons):
        """
        Add readout noise to the number of electrons.
        :param n_electrons: number of electrons
        :return: number of electrons with readout noise
        """
        image_size = n_electrons.shape
        readout_noise = np.floor(np.random.normal(0, self.std_read, image_size))
        return n_electrons + readout_noise

    def _dark_current_noise(self, n_electrons):
        """
        Add dark current noise to the number of electrons.
        :param n_electrons: number of electrons
        :return: number of electrons with dark current noise
        """
        dark_current = np.floor(
            np.random.normal(self.beta_t * self.exposure_time, np.sqrt(self.beta_t * self.exposure_time), n_electrons.shape))
        return n_electrons + dark_current

    def _quantization_noise(self, n_electrons):
        """
        Add full well noise to the number of electrons.
        :param n_electrons: number of electrons
        :return: number of electrons with full well noise
        """
        full_well_factor = 2 ** self.n_bits / self.full_well
        n_electrons_full_well = 1 / full_well_factor * np.floor(full_well_factor * n_electrons)
        return n_electrons_full_well
