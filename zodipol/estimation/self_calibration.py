import numpy as np
import astropy.units as u

from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.sparse import eye
from tqdm import tqdm

from zodipol.zodipol import Observation
from zodipol.estimation.estimate_signal import estimate_IQU
from zodipol.mie_scattering.mueller_matrices import get_rotation_mueller_matrix
from zodipol.zodipy_local.zodipy.zodipy import IQU_to_image
from zodipol.utils.math import ang2vec


class SelfCalibration:
    def __init__(self, images_res_flat, rotation_list, zodipol, parser, theta=None, phi=None):
        self.obs = None
        self.images = images_res_flat
        self.rotation_list = rotation_list
        self.zodipol = zodipol
        self.parser = parser
        self.theta = theta
        self.phi = phi
        self.nobs = len(self.rotation_list)
        self.aligned_images = self.align_images(images_res_flat.value, rotation_list, fill_value=np.nan)
        self.initialize()

    def initialize(self, init = None):
        self.p = np.ones(self.parser["resolution"]).reshape((-1))
        self.eta = np.zeros(self.parser["resolution"]).reshape((-1))
        self.delta = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)
        self.alpha = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)
        if init is not None and 'p' in init:
            self.p = init['p']
        if init is not None and 'eta' in init:
            self.eta = init['eta']
        if init is not None and 'delta' in init:
            self.delta = init['delta']
        if init is not None and 'alpha' in init:
            self.alpha = init['alpha']

    def get_properties(self):
        nan_ind = np.isnan(self.aligned_images).any(axis=-2).any(axis=-1, keepdims=True).squeeze()
        p_nan = np.where(nan_ind, np.nan, self.p)
        eta_nan = np.where(nan_ind, np.nan, self.eta)
        delta_nan, alpha_nan = np.where(nan_ind, np.nan, self.delta), np.where(nan_ind, np.nan, self.alpha)
        return p_nan, eta_nan, delta_nan, alpha_nan

    def forward_model(self, obs=None, p=None, eta=None, delta=None, alpha=None):
        obs = (obs if obs is not None else self.obs)
        p = (p if p is not None else self.p)
        eta = (eta if eta is not None else self.eta)
        delta = (delta if delta is not None else self.delta)
        alpha = (alpha if alpha is not None else self.alpha)

        forward_results = []
        for ii in range(len(obs)):
            cur_p = p[:, None]
            cur_eta = eta[:, None] + self.parser["polarization_angle"]
            biref_mueller = self.zodipol.imager.get_birefringence_mueller_matrix(delta, alpha)
            biref_obs = self.zodipol.imager.apply_birefringence(obs[ii], biref_mueller)

            I, Q, U = biref_obs.I, biref_obs.Q, biref_obs.U
            img_model = IQU_to_image(I, Q, U, cur_p, cur_eta)
            forward_results.append(img_model)
        results = np.stack(forward_results, axis=-1)
        return results

    def calibrate(self, n_itr=20, mode="all", disable=False, callback=None, init=None):
        self.initialize(init)
        itr_cost = []
        itr_callback = []
        for _ in tqdm(range(n_itr), disable=disable):
            self._calibrate_itr(mode=mode)
            itr_cost.append(self.get_mse())
            if callback is not None:
                itr_callback.append(callback(self))
        if callback is not None:
            return itr_cost, itr_callback
        return itr_cost

    def get_mse(self):
        img_model = self.forward_model()
        mse = np.nanmean((img_model - self.images.value) ** 2)
        return mse

    def _calibrate_itr(self, mode="all"):
        self.obs = self.estimate_observations()
        if mode == "all" or 'P' in mode:
            self.p = self._calibrate_property("p")
        if mode == "all" or 'eta' in mode:
            self.eta = self._calibrate_property("eta")
        if mode == "all" or 'delta' in mode:
            self.delta = self._calibrate_property("delta")
        if mode == "all" or 'alpha' in mode:
            self.alpha = self._calibrate_property("alpha")
        if mode != "all" and not any([x in mode for x in ['P', 'eta', 'delta', 'alpha']]):
            raise ValueError(f"Self-calibration mode {mode} not recognized")

    def _calibrate_property(self, property_name):
        def cost_function(x):
            x = x.reshape(getattr(self, property_name).shape)
            img = self.forward_model(**{property_name: x})
            diff_resh = img - self.images.value
            cost = 1e20 * (diff_resh ** 2).sum(axis=(-1, -2)) ** 0.5
            cost = np.nan_to_num(cost, nan=0)  # zero out nans
            return cost

        propsize = getattr(self, property_name).size
        jac_sparsity = eye(propsize)
        x0 = getattr(self, property_name).flatten()
        bounds = self.get_property_bounds(property_name)
        p_lsq = least_squares(cost_function, x0=x0.flatten(), jac_sparsity=jac_sparsity, ftol=1e-6, max_nfev=30,
                              bounds=bounds, verbose=2)
        return p_lsq.x.reshape(getattr(self, property_name).shape)

    @staticmethod
    def get_property_bounds(property_name):
        if property_name == "p":
            return 0.5, 1
        elif property_name == "eta":
            return -np.pi/4, np.pi/4
        elif property_name == "delta":
            return 0, np.pi/2
        elif property_name == "alpha":
            return -np.pi, np.pi
        else:
            raise ValueError("Unknown property name")

    def estimate_observations(self):
        rotation_list = self.rotation_list
        # interp_images_res = self.zodipol.post_process_images(self.aligned_images, sign=-1)
        interp_images_res = self.aligned_images
        # est_shape = (interp_images_res.shape[0], np.prod(interp_images_res.shape[1:]))
        p = self.align_images(self.p[:, None].repeat(self.nobs, axis=-1), rotation_list, invert=True)[..., None, :].repeat(self.parser["n_polarization_ang"], axis=-2)
        eta = self.align_images(self.eta[:, None, None].repeat(self.nobs, axis=-1), rotation_list, invert=True) + self.parser["polarization_angle"][:, None]
        mueller = self.zodipol.imager.get_birefringence_mueller_matrix(self.delta, self.alpha)
        mueller_rot = mueller[..., None].repeat(self.nobs, axis=-1)
        rotation_mat = get_rotation_mueller_matrix(-np.deg2rad(rotation_list))[None, ...].repeat(self.images.shape[0], axis=0)
        I_est, Q_est, U_est = self.estimate_IQU(interp_images_res, rotation_mat, p, eta, mueller_rot)
        obs_est = Observation(I_est, Q_est, U_est, theta=self.theta, phi=self.phi, roll=0)
        obs_est_rot = [self.realign_observation(obs_est, roll) for roll in rotation_list]
        return obs_est_rot

    @staticmethod
    def estimate_IQU(intensity, rotation_mat, polarizance, angles, mueller):
        """
        Calculate the Stokes parameters of the signal.
        :param intensity: The intensity of the signal.
        :param angles: The angles of the signal.
        :return: The Stokes parameters of the signal.
        """
        # create the angles matrix
        n_pixels = intensity.shape[0]
        angles_matrix = 0.5 * np.stack((np.ones_like(angles), polarizance*np.cos(2*angles), polarizance*np.sin(2*angles)), axis=-3)
        forward_mat = np.einsum('...jiw,...jkw,...wks->...siw', angles_matrix, mueller[:, :3, :3, :], rotation_mat[..., :3, :3])
        forward_mat_s, intensity_s = forward_mat.reshape((n_pixels, 3, -1)), intensity.reshape((n_pixels, -1))
        # forward_mat_s, intensity_s = np.nan_to_num(forward_mat_s, nan=0), np.nan_to_num(intensity_s, nan=0)

        pseudo_inverse = np.einsum('...ij,...kj->...ik', forward_mat_s, forward_mat_s)
        intensity_mult = np.einsum('...ij,...j->...i', forward_mat_s, intensity_s)
        intensity_inv = np.linalg.solve(pseudo_inverse, intensity_mult)
        I, Q, U = intensity_inv[..., 0], intensity_inv[..., 1], intensity_inv[..., 2]
        return I, Q, U

    def align_images(self, images_res, rotation_arr, invert=False, fill_value=0):
        if invert:
            rotation_arr = rotation_arr
        res_images = []
        for ii in range(len(rotation_arr)):
            rot_image = self.get_rotated_image(images_res[..., ii], -rotation_arr[ii], fill_value=fill_value)
            res_images.append(rot_image)
        return np.stack(res_images, axis=-1)

    def realign_observation(self, obs, roll):
        I_interp = self.get_rotated_image(obs.I, roll)
        Q_interp = self.get_rotated_image(obs.Q, roll)
        U_interp = self.get_rotated_image(obs.U, roll)
        obs_new = Observation(I_interp, Q_interp, U_interp, theta=obs.theta, phi=obs.phi, roll=0)
        return obs_new.change_roll(np.deg2rad(roll))

    def get_rotated_image(self, images, rotation_to, fill_value=0):
        if rotation_to == 0:  # avoid interpolation issues
            return images
        theta_from, phi_from = self.zodipol.create_sky_coords(theta=self.parser["direction"][0], phi=self.parser["direction"][1],
                                                              roll=0 * u.deg, resolution=self.parser["resolution"])
        theta_to, phi_to = self.zodipol.create_sky_coords(theta=self.parser["direction"][0], phi=self.parser["direction"][1],
                                                          roll=rotation_to * u.deg, resolution=self.parser["resolution"])
        vec_from = ang2vec(theta_from, phi_from)
        vec_to = ang2vec(theta_to, phi_to)
        x = np.linspace(vec_from[:, 0].min(), vec_from[:, 0].max(), self.parser["resolution"][1])
        y = np.linspace(vec_from[:, 1].min(), vec_from[:, 1].max(), self.parser["resolution"][0])

        images_resh = images.reshape(self.parser["resolution"] + list(images.shape[1:]))
        interp_rg = RegularGridInterpolator((y, x), images_resh, bounds_error=False, fill_value=fill_value)

        interp = interp_rg(list(zip(vec_to[:, 1], vec_to[:, 0])))
        return interp
    