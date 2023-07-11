import os
import numpy as np
import astropy.units as u
import healpy as hp

from astropy.time import Time
from astropy.constants import c

from zodipol.utils.math import ang2vec, vec2ang, get_c2w
from zodipol.zodipol.observation import Observation
from zodipol.zodipy_local.zodipy.zodipy import Zodipy, IQU_to_image
from zodipol.imager.imager import Imager
from zodipol.background_radiation import IntegratedStarlight, PlanetaryLight, IntegratedStarlightFactory


MIE_MODEL_DEFAULT_PATH = 'saved_models/white_light_mie_model.npz'
INTEGRATED_STARLIGHT_MODEL_PATH = 'saved_models/skymap_flux.npz'


class Zodipol:
    def __init__(self, polarizance: float = 1., fov: float = None,  n_polarization_ang: int = 4, solar_cut=30 * u.deg,
                 parallel=False, n_freq: int = 5, isl=True,
                 integrated_starlight_path=INTEGRATED_STARLIGHT_MODEL_PATH, planetary: bool = True, resolution=(2448, 2048),
                 imager_params=None, zodipy_params=None, color='red'):
        imager_params = (imager_params if imager_params is not None else {})
        zodipy_params = (zodipy_params if zodipy_params is not None else {})
        self.polarization_angle = np.linspace(0, np.pi, n_polarization_ang, endpoint=False)  # Polarization angle of the observation
        self.polarizance = polarizance  # Polarizance of the observation
        self.fov = fov  # Field of view (deg)

        self.imager = Imager(resolution=resolution, **imager_params)  # Initialize the imager
        self._set_imager_spectrum(n_freq=n_freq, color=color)  # Generate the spectrum

        self.zodipy = Zodipy("dirbe", solar_cut=solar_cut, extrapolate=True, parallel=parallel, **zodipy_params)  # Initialize the model

        self.isl = (self._set_integrated_starlight(integrated_starlight_path=integrated_starlight_path) if isl is True else None)  # Initialize the integrated starlight model
        self.planetary = (PlanetaryLight() if planetary is True else None)

    def create_observation(self, theta: u.Quantity, phi: u.Quantity, roll: u.Quantity = 0 * u.deg,
                           obs_time: Time | str = Time("2022-06-14"), lonlat: bool = False, new_isl=False):
        obs_time = (Time(obs_time) if not isinstance(obs_time, Time) else obs_time)  # Observation time
        theta_vec, phi_vec = self.create_sky_coords(theta=theta, phi=phi, roll=roll, resolution=self.imager.resolution)
        binned_emission = self.zodipy.get_emission_ang(self.frequency, theta_vec, phi_vec,
                                                              obs_time=obs_time,
                                                              weights=self.frequency_weight, lonlat=lonlat,
                                                              polarization_angle=self.polarization_angle,
                                                              polarizance=self.polarizance, return_IQU=True)
        isl_map = self._get_integrated_starlight_ang(theta_vec, phi_vec, new_isl=new_isl, width=self.fov / min(self.imager.resolution))
        planets_skymap = self._get_planetary_light_ang(theta_vec, phi_vec, obs_time)
        I, Q, U = binned_emission[..., 0], binned_emission[..., 1], binned_emission[..., 2]
        I += planets_skymap + isl_map
        if isinstance((isl_map + planets_skymap), u.Quantity):
            star_pixels = ((isl_map + planets_skymap) > 1e-21 * u.Unit('W / m^2 sr Hz')).any(axis=1)
        else:
            star_pixels = None
        return Observation(I, Q, U, theta=theta_vec, phi=phi_vec, roll=0, star_pixels=star_pixels).change_roll(roll.to('rad').value)

    def create_full_sky_observation(self, nside: int = 64, obs_time: Time | str = Time("2022-06-14"),):
        obs_time = (Time(obs_time) if not isinstance(obs_time, Time) else obs_time)  # Observation time
        binned_emission = self.zodipy.get_binned_emission_pix(self.frequency, np.arange(hp.nside2npix(nside)), nside,
                                                              obs_time=obs_time,
                                                              weights=self.frequency_weight,
                                                              polarization_angle=self.polarization_angle,
                                                              polarizance=self.polarizance, return_IQU=True)
        isl_map = self._get_integrated_starlight_map(nside)
        planets_skymap = self._get_planetary_light_map(nside, obs_time)
        I, Q, U = binned_emission[..., 0], binned_emission[..., 1], binned_emission[..., 2]
        I += planets_skymap + isl_map
        if isinstance((isl_map + planets_skymap), u.Quantity):
            star_pixels = ((isl_map + planets_skymap) > 1e-21 * u.Unit('W / m^2 sr Hz')).any(axis=1)
        else:
            star_pixels = None
        return Observation(I, Q, U, star_pixels=star_pixels)

    def make_camera_images_multicolor(self, *args, **kwargs):
        colors = self.imager.get_pixel_colors()
        images = []
        for color in colors:
            images.append(self.make_camera_images(*args, color=color, **kwargs))
        return np.stack(images, axis=-2)

    def post_process_images(self, images, sign=-1, color='red'):
        imager_response = self.get_imager_response(color=color)
        dark_current_electrons = self.imager.camera_dark_current_estimation()
        dark_current = self.imager.number_of_electrons_to_intensity(dark_current_electrons, self.frequency, imager_response)
        images += sign*dark_current.squeeze()
        return images

    def make_camera_images(self, obs: Observation, polarizance=None, polarization_angle=None, add_noise=True,
                           n_realizations=1, fillna=None, noise_params: dict = None, color='red'):
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
        imager_response = self.get_imager_response(color=color)

        binned_emission_real = IQU_to_image(obs.I, obs.Q, obs.U, polarizance, polarization_angle)
        n_electrons_real = self.imager.intensity_to_number_of_electrons(binned_emission_real, frequency=self.frequency,
                                                                        weights=imager_response)

        n_electrons_real_rpt = n_electrons_real[..., None].repeat(n_realizations, axis=-1)
        if add_noise:
            n_electrons_real_rpt = self.imager.imager_noise_model(n_electrons_real_rpt, **noise_params)
        else:
            n_electrons_real_rpt += self.imager.camera_dark_current_estimation()
            n_electrons_real_rpt = self.imager.camera_post_process(n_electrons_real_rpt)
        camera_intensity_real = self.imager.number_of_electrons_to_intensity(n_electrons_real_rpt, self.frequency, imager_response)
        if fillna is not None:
            camera_intensity_real = np.nan_to_num(camera_intensity_real, nan=fillna * camera_intensity_real.unit)
        camera_intensity_mean = camera_intensity_real.mean(axis=-1)
        return camera_intensity_mean

    def combine_observations(self, obs_list, polarizance=None, polarization_angle=None, fillna=None, color='red'):
        polarizance = polarizance if polarizance is not None else self.polarizance
        polarization_angle = polarization_angle[:,None]+self.polarization_angle[None, :] if polarization_angle is not None else self.polarization_angle
        imager_response = self.get_imager_response(color=color)
        obs_res = []
        for obs in obs_list:  # take multiple relatizations of the projection
            binned_emission_real = IQU_to_image(obs.I, obs.Q, obs.U, polarizance[:, None, :], polarization_angle[..., None, :])
            n_electrons_real = self.imager.intensity_to_number_of_electrons(binned_emission_real, frequency=self.frequency, weights=imager_response)
            camera_intensity_real = self.imager.number_of_electrons_to_intensity(n_electrons_real, self.frequency, imager_response)
            if fillna is not None:
                camera_intensity_real = np.nan_to_num(camera_intensity_real, nan=fillna * camera_intensity_real.unit)
            obs_combine = Observation.from_image(camera_intensity_real, polarizance, polarization_angle,
                                                 theta=obs.theta, phi=obs.phi, roll=obs.roll, star_pixels=obs.star_pixels)
            obs_res.append(obs_combine)
        return obs_res


    def create_sky_coords(self, theta: u.Quantity, phi: u.Quantity, roll: u.Quantity = 0 * u.deg, resolution=(2448, 2048)):
        theta_vec = np.linspace(- self.fov / 2, self.fov / 2, resolution[1]) + 90*u.deg
        phi_vec = np.linspace(- self.fov / 2, self.fov / 2, resolution[0])
        theta_mat, phi_mat = np.meshgrid(theta_vec, phi_vec)

        trans_mat = get_c2w(theta.to('rad').value, phi.to('rad').value, roll.to('rad').value)

        rot_vec = ang2vec(theta_mat.to('rad').value, phi_mat.to('rad').value)
        mul = np.einsum('ij,...j->...i', trans_mat, rot_vec)
        theta_v, phi_v = vec2ang(mul) * u.rad
        return theta_v.flatten(), phi_v.flatten()

    # def _set_mie_model(self, mie_model_path=MIE_MODEL_DEFAULT_PATH):
    #     if mie_model_path is None:
    #         return  # No mie model
    #     if os.path.isfile(mie_model_path):
    #         mie_model = MieScatteringModel.load(mie_model_path)
    #     else:
    #         spectrum = np.logspace(np.log10(300), np.log10(700), 20)  # white light wavelength in nm
    #         mie_model = MieScatteringModel.train(spectrum)
    #     self.mie_model = mie_model

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
        jacobian = c / (self.frequency ** 2)
        isl_map = isl_map * jacobian[None, ...]
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

        jacobian = c / (self.frequency ** 2)
        isl_map = isl_map * jacobian[None, ...]
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

    def _set_imager_spectrum(self, n_freq=5, color='red'):
        self.wavelength, self.frequency = self._generate_spectrum(n_freq=n_freq, color=color)
        self.frequency_weight = np.ones_like(self.frequency)  # Weight of the frequencies

    def get_imager_response(self, color='red'):
        return self.imager.get_camera_response(self.wavelength.value, color)

    def _generate_spectrum(self, n_freq=5, color='red'):
        f_min, f_max = 430 * u.THz, 995 * u.THz  # Frequency of the observation
        frequency = np.linspace(f_min, f_max, n_freq, endpoint=True)
        wavelength_interp = frequency.to(u.nm, equivalencies=u.spectral())
        return wavelength_interp, frequency

