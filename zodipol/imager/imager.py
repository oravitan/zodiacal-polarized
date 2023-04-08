import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import h, c
from scipy.stats import multivariate_normal

from zodipol.zodipol.observation import Observation
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
                 resolution=(2448, 2048)):
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
        self.resolution = resolution

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
        if channel == 'red' or channel == 'r':
            return np.interp(wavelength, IMAGER_RESPONSE_RED.index, IMAGER_RESPONSE_RED['<Efficiency>'])
        elif channel == 'green' or channel == 'g':
            return np.interp(wavelength, IMAGER_RESPONSE_GREEN.index, IMAGER_RESPONSE_GREEN['<Efficiency>'])
        elif channel == 'blue' or channel == 'b':
            return np.interp(wavelength, IMAGER_RESPONSE_BLUE.index, IMAGER_RESPONSE_BLUE['<Efficiency>'])
        else:
            raise ValueError('Channel must be either red, green or blue')

    def get_wavelength_range(self, channel: str):
        if channel == 'red' or channel == 'r':
            return IMAGER_RESPONSE_RED.index
        elif channel == 'green' or channel == 'g':
            return IMAGER_RESPONSE_GREEN.index
        elif channel == 'blue' or channel == 'b':
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
            df = np.gradient(frequency)
            n_electrons = np.einsum('ij...,j->i...', energy, df / (h * frequency) * weights)  # Number of electrons per frequency
        else:
            frequency = wavelength.to(u.THz, equivalencies=u.spectral())
            df = np.gradient(frequency)
            jacobian = - c / (wavelength ** 2)
            n_electrons = np.einsum('ij...,j->i...', energy, jacobian * df / (h * frequency) * weights)

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
        actual mean is self.beta_t * self.exposure_time, but we assume it's accounted for
        :param n_electrons: number of electrons
        :return: number of electrons with dark current noise
        """
        center = self.camera_dark_current_estimation()
        dark_current = np.floor(np.random.normal(center, (self.beta_t * self.exposure_time)**0.25, n_electrons.shape))
        return n_electrons + dark_current

    def _quantization_noise(self, n_electrons):
        """
        Add full well noise to the number of electrons.
        :param n_electrons: number of electrons
        :return: number of electrons with full well noise
        """
        full_well_factor = (2 ** self.n_bits) / self.full_well
        n_electrons_full_well = (1 / full_well_factor) * np.round(full_well_factor * n_electrons)
        return n_electrons_full_well

    # ------------------------------------------------
    # ------------- Birefringence model --------------
    # ------------------------------------------------
    def apply_birefringence(self, obs, biref_mat):
        observation_mat = obs.to_numpy()
        observation_biref = np.einsum('...ij,...jk->...ik', biref_mat, observation_mat[..., None])
        I, Q, U = observation_biref[..., 0, 0], observation_biref[..., 1, 0], observation_biref[..., 2, 0]
        return Observation(I, Q, U, theta=obs.theta, phi=obs.phi)

    def get_birefringence_mueller_matrix(self, birefringence_amount, birefringence_angle):
        pol_mueller = self._get_birefringence_polarizer_mat(birefringence_amount)
        pol_rotation = self._get_birefringence_angle_rotation_mat(birefringence_angle)
        pol_rotation_inv = self._get_birefringence_angle_rotation_mat(-birefringence_angle)
        mul1 = np.einsum('...ij,...jk->...ik', pol_mueller, pol_rotation)
        mul2 = np.einsum('...ij,...jk->...ik', pol_rotation_inv, mul1)
        return mul2

    def _get_birefringence_angle_rotation_mat(self, birefringence_angle):
        c2 = np.cos(2 * birefringence_angle)
        s2 = np.sin(2 * birefringence_angle)
        mueller = np.zeros(birefringence_angle.shape + (4, 4))
        mueller[..., 0, 0] = 1
        mueller[..., 1, 1] = c2
        mueller[..., 1, 2] = -s2
        mueller[..., 2, 1] = s2
        mueller[..., 2, 2] = c2
        mueller[..., 3, 3] = 1
        return mueller

    def _get_birefringence_polarizer_mat(self, birefringence_amount):
        c = np.cos(birefringence_amount)
        s = np.sin(birefringence_amount)
        mueller = np.zeros(birefringence_amount.shape + (4, 4))
        mueller[..., 0, 0] = 1
        mueller[..., 1, 1] = 1
        mueller[..., 2, 2] = c
        mueller[..., 2, 3] = s
        mueller[..., 3, 2] = -s
        mueller[..., 3, 3] = c
        return mueller

    def get_birefringence_mat(self, value=0.5, type='constant', flat=False, **kwargs):
        """
        Calculate the birefringence amount per-pixel
        :return: birefringence amount
        """
        if type == 'constant':
            biref_value = np.ones(self.resolution) * value
        elif type == 'center':
            std = (kwargs['std'] if 'std' in kwargs else 3) * np.ones((2,))
            center = np.array((0.5, 0.5))
            norm = multivariate_normal(mean=center, cov=np.diag(std))
            x, y = np.meshgrid(*(np.linspace(0, 1, r) for r in self.resolution))
            biref_value = norm.pdf(np.dstack((x.T, y.T)))
            biref_value = value * (biref_value - biref_value.min()) / (biref_value.max() - biref_value.min())
            if 'inv' in kwargs and kwargs['inv']:
                biref_value = value - biref_value
        elif type == 'linear':
            angle = kwargs["angle"] if "angle" in kwargs else 0
            min_v = kwargs["min"] if "min" in kwargs else -value
            x, y = np.meshgrid(np.linspace(min_v, value, self.resolution[0]),
                               np.linspace(min_v, value, self.resolution[1]), indexing='ij')
            biref_value = x * np.cos(angle) - y * np.sin(angle)
        elif type == 'sine':
            raise NotImplementedError('Center birefringence is not implemented yet.')
        else:
            raise ValueError(f'Birefringence type \"{type}\"is not a valid type.')

        if flat:
            return biref_value.flatten()
        return biref_value

