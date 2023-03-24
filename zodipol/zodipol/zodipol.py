import os
import numpy as np
import astropy.units as u
import healpy as hp

from astropy.time import Time
from skimage.filters import gaussian

from zodipol.zodipol.observation import Observation
from zodipol.mie_scattering.mie_scattering_model import MieScatteringModel
from zodipol.zodipy_local.zodipy.zodipy import Zodipy, IQU_to_image
from zodipol.utils.math import get_rotation_matrix
from zodipol.imager.imager import Imager
from zodipol.background_radiation import IntegratedStarlight, PlanetaryLight, IntegratedStarlightFactory


MIE_MODEL_DEFAULT_PATH = 'saved_models/white_light_mie_model.npz'
INTEGRATED_STARLIGHT_MODEL_PATH = 'saved_models/skymap_flux.npz'


class Zodipol:
    def __init__(self, polarizance: float = 1., fov: float = None,  n_polarization_ang: int = 4, solar_cut=30 * u.deg,
                 parallel=False, n_freq: int = 5, mie_model_path=MIE_MODEL_DEFAULT_PATH, isl=True,
                 integrated_starlight_path=INTEGRATED_STARLIGHT_MODEL_PATH, planetary: bool = True, resolution=(2448, 2048),
                 imager_params=None, zodipy_params=None, direction_uncertainty=0*u.deg, circ_motion_blur = 0*u.deg):
        imager_params = (imager_params if imager_params is not None else {})
        zodipy_params = (zodipy_params if zodipy_params is not None else {})
        self.polarization_angle = np.linspace(0, np.pi, n_polarization_ang, endpoint=False)  # Polarization angle of the observation
        self.polarizance = polarizance  # Polarizance of the observation
        self.fov = fov  # Field of view (deg)

        self.imager = Imager(resolution=resolution, **imager_params)  # Initialize the imager
        self._set_imager_spectrum(n_freq=n_freq)  # Generate the spectrum

        self.zodipy = Zodipy("dirbe", solar_cut=solar_cut, extrapolate=True, parallel=parallel, **zodipy_params)  # Initialize the model
        self._set_mie_model(mie_model_path=mie_model_path)  # Initialize the mie model

        self.isl = (self._set_integrated_starlight(integrated_starlight_path=integrated_starlight_path) if isl is True else None)  # Initialize the integrated starlight model
        self.planetary = (PlanetaryLight() if planetary is True else None)
        self.direction_uncertainty = direction_uncertainty
        self.circ_motion_blur = circ_motion_blur

    def create_observation(self, theta: u.Quantity, phi: u.Quantity, roll: u.Quantity = 0 * u.deg,
                           obs_time: Time | str = Time("2022-06-14"), lonlat: bool = False, new_isl=False):
        obs_time = (Time(obs_time) if not isinstance(obs_time, Time) else obs_time)  # Observation time
        theta_vec, phi_vec = self._create_sky_coords(theta=theta, phi=phi, roll=roll, resolution=self.imager.resolution)
        binned_emission = self.zodipy.get_emission_ang(self.frequency, theta_vec, phi_vec,
                                                              obs_time=obs_time, mie_scattering_model=self.mie_model,
                                                              weights=self.frequency_weight, lonlat=lonlat,
                                                              polarization_angle=self.polarization_angle,
                                                              polarizance=self.polarizance, return_IQU=True)
        isl_map = self._get_integrated_starlight_ang(theta_vec, phi_vec, new_isl=new_isl, width=self.fov / min(self.imager.resolution))
        planets_skymap = self._get_planetary_light_ang(theta_vec, phi_vec, obs_time)
        I, Q, U = binned_emission[..., 0], binned_emission[..., 1], binned_emission[..., 2]
        I += planets_skymap + isl_map
        I, Q, U = self._add_direction_uncertainty(I, Q, U)
        I, Q, U = self._add_radial_blur(I, Q, U)
        return Observation(I, Q, U, theta=theta_vec, phi=phi_vec, roll=0).change_roll(roll.to('rad').value)

    def create_full_sky_observation(self, nside: int = 64, obs_time: Time | str = Time("2022-06-14"),):
        obs_time = (Time(obs_time) if not isinstance(obs_time, Time) else obs_time)  # Observation time
        binned_emission = self.zodipy.get_binned_emission_pix(self.frequency, np.arange(hp.nside2npix(nside)), nside,
                                                              obs_time=obs_time, mie_scattering_model=self.mie_model,
                                                              weights=self.frequency_weight,
                                                              polarization_angle=self.polarization_angle,
                                                              polarizance=self.polarizance, return_IQU=True)
        isl_map = self._get_integrated_starlight_map(nside)
        planets_skymap = self._get_planetary_light_map(nside, obs_time)
        I, Q, U = binned_emission[..., 0], binned_emission[..., 1], binned_emission[..., 2]
        I += planets_skymap + isl_map
        return Observation(I, Q, U)

    def make_camera_images(self, obs: Observation, polarizance=None, polarization_angle=None, add_noise=True,
                           n_realizations=1, fillna=None, noise_params: dict = None):
        """
        Make a camera image from frequency-dependent the observation
        :param obs: Observation of the sky
        :param polarizance: Polarizance of the camera
        :param polarization_angle: Polarization angle of the camera
        :param add_noise: Add noise to the image
        :param n_realizations: Number of realizations of the image
        :return: Combined camera image
        """
        polarizance = polarizance if polarizance is not None else self.polarizance
        polarization_angle = polarization_angle if polarization_angle is not None else self.polarization_angle
        noise_params = (noise_params if noise_params is not None else {})
        realization_list = []  # List of the realizations
        for ii in range(n_realizations):  # take multiple relatizations of the projection
            binned_emission_real = IQU_to_image(obs.I, obs.Q, obs.U, polarizance, polarization_angle)

            # Calculate the number of photons
            n_electrons_real = self.imager.intensity_to_number_of_electrons(binned_emission_real, frequency=self.frequency, weights=self.imager_response)
            if add_noise:  # Add noise to the image
                n_electrons_real = self.imager.imager_noise_model(n_electrons_real, **noise_params)
            else:  # assume no image noise
                n_electrons_real += self.imager.camera_dark_current_estimation()
                n_electrons_real = self.imager.camera_post_process(n_electrons_real)
            camera_intensity_real = self.imager.number_of_electrons_to_intensity(n_electrons_real, self.frequency, self.imager_response)
            if fillna is not None:
                camera_intensity_real = np.nan_to_num(camera_intensity_real, nan=fillna * camera_intensity_real.unit)
            realization_list.append(camera_intensity_real)
        camera_intensity_mean = np.stack(realization_list, axis=-1).mean(axis=-1)  # Mean of the realizations
        return camera_intensity_mean

    def _create_sky_coords(self, theta: u.Quantity, phi: u.Quantity, roll: u.Quantity = 0*u.deg, resolution=(2448, 2048)):
        # Create the sky coordinates
        theta_vec = np.linspace(- self.fov / 2, self.fov / 2, resolution[1])
        phi_vec = np.linspace(- self.fov / 2, self.fov / 2, resolution[0])
        theta_mat, phi_mat = np.meshgrid(theta + theta_vec, phi + phi_vec)
        theta_v, phi_v = theta_mat.flatten(), phi_mat.flatten()
        theta_v = abs(180 * u.deg - abs(180 * u.deg - theta_v))  # Flip the theta axis to the correct direction

        # Rotate the coordinates according to the roll angle
        rot_vec = hp.ang2vec(theta.to("rad"), phi.to("rad"))
        vecs = hp.ang2vec(theta_v.to('rad'), phi_v.to('rad'))
        vecs_rot = (get_rotation_matrix(rot_vec, roll) @ vecs.T).T
        theta_v, phi_v = hp.vec2ang(vecs_rot.value) * u.rad

        return theta_v, phi_v

    def _set_mie_model(self, mie_model_path=MIE_MODEL_DEFAULT_PATH):
        if mie_model_path is None:
            return  # No mie model
        if os.path.isfile(mie_model_path):
            mie_model = MieScatteringModel.load(mie_model_path)
        else:
            spectrum = np.logspace(np.log10(300), np.log10(700), 20)  # white light wavelength in nm
            mie_model = MieScatteringModel.train(spectrum)
        self.mie_model = mie_model

    def _set_integrated_starlight(self, integrated_starlight_path=INTEGRATED_STARLIGHT_MODEL_PATH):
        if integrated_starlight_path is None or not os.path.isfile(integrated_starlight_path):
            return
        isl = IntegratedStarlight.load(integrated_starlight_path)
        isl.interpolate_freq(self.frequency.to('Hz'), update=True)
        return isl

    def _get_integrated_starlight_map(self, nside: int):
        if self.isl is None:
            return 0
        isl_map = self.isl.resize_skymap(nside)
        df, dw = np.gradient(self.frequency), -np.gradient(self.wavelength)
        isl_map = isl_map * (dw / df)[None, ...]
        # isl_map = np.stack([isl_map] * len(self.polarization_angle), axis=-1)
        return isl_map

    def _get_integrated_starlight_ang(self, theta: u.Quantity, phi: u.Quantity, new_isl=False, width=None):
        if self.isl is None and not new_isl:
            return 0
        elif new_isl:
            isf = IntegratedStarlightFactory()
            isl = isf.build_dirmap(theta, phi, self.frequency.to('Hz').value, width, parallel=False)
            isl_map = isl.isl_map
        else:
            isl = self.isl
            isl_map = isl.get_ang(theta.to('rad').value, phi.to('rad').value)  # resample

        df, dw = np.gradient(self.frequency), -np.gradient(self.wavelength)
        isl_map = isl_map * (dw / df)[None, ...]
        # isl_map = np.stack([isl_map] * len(self.polarization_angle), axis=-1)
        return isl_map

    def _get_planetary_light_map(self, nside: int, obs_time):
        if self.planetary is None:
            return 0
        planets_skymap = self.planetary.make_planets_map(nside, obs_time, self.wavelength)
        # planets_skymap = np.stack([planets_skymap] * len(self.polarization_angle), axis=-1)
        return planets_skymap

    def _get_planetary_light_ang(self, theta: u.Quantity, phi: u.Quantity, obs_time):
        if self.planetary is None:
            return 0
        planets_skymap = self.planetary.get_ang(obs_time, theta.to('rad').value, phi.to('rad').value, self.wavelength)
        # planets_skymap = np.stack([planets_skymap] * len(self.polarization_angle), axis=-1)
        return planets_skymap

    def _set_imager_spectrum(self, n_freq=5):
        self.wavelength, self.frequency, self.imager_response = self._generate_spectrum(n_freq=n_freq)
        self.frequency_weight = np.ones_like(self.frequency)  # Weight of the frequencies

    def _generate_spectrum(self, n_freq=5):
        wavelength_range = self.imager.get_wavelength_range('red').values * u.nm
        frequency_range = wavelength_range.to(u.THz, equivalencies=u.spectral())  # Frequency of the observation
        imager_response = self.imager.get_camera_response(wavelength_range.value, 'red')

        f_min = max(frequency_range.min(), 440 * u.THz)
        f_max = min(frequency_range.max(), 530 * u.THz)
        frequency = np.linspace(f_min, f_max, n_freq, endpoint=True)
        wavelength_interp = frequency.to(u.nm, equivalencies=u.spectral())
        imager_response_interp = np.interp(wavelength_interp, wavelength_range, imager_response)

        return wavelength_interp, frequency, imager_response_interp

    def _add_direction_uncertainty(self, I, Q, U):
        pixel_size = self.fov / self.imager.resolution
        pixels_uncertainty = (self.direction_uncertainty / pixel_size).value
        pixels_uncertainty = np.concatenate((pixels_uncertainty, [1]))
        prev_shape = I.shape
        new_shape = self.imager.resolution + list(I.shape[1:])
        I_g, Q_g, U_g = [gaussian(x.reshape(new_shape), pixels_uncertainty).reshape(prev_shape) for x in (I, Q, U)]
        I_g, Q_g, U_g = [x * I.unit for x in (I_g, Q_g, U_g)]
        return I_g, Q_g, U_g

    def _add_radial_blur(self, I, Q, U):
        if self.circ_motion_blur.value != 0:
            raise NotImplementedError
        return I, Q, U