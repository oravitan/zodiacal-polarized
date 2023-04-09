import numpy as np
import astropy.units as u

from scipy.interpolate import griddata
from scipy.optimize import least_squares
from scipy.sparse import eye
from tqdm import tqdm

from zodipol.zodipol import Observation
from zodipol.estimation.estimate_signal import estimate_IQU
from zodipol.zodipy_local.zodipy.zodipy import IQU_to_image


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
        self.aligned_images = self.align_images(images_res_flat.value, rotation_list)
        self.initialize()

    def initialize(self):
        self.p = np.ones(self.parser["resolution"]).reshape((-1, 1)).repeat(self.nobs, axis=-1)
        self.eta = np.zeros(self.parser["resolution"]).reshape((-1, 1)).repeat(self.nobs, axis=-1)
        self.delta = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)[..., None].repeat(self.nobs, axis=-1)
        self.alpha = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)[..., None].repeat(self.nobs, axis=-1)

    def get_properties(self):
        nan_ind = np.isnan(self.aligned_images).any(axis=-2).any(axis=-1, keepdims=True)
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
            cur_p = p[..., ii, None]
            cur_eta = eta[..., ii, None] + self.parser["polarization_angle"]
            biref_mueller = self.zodipol.imager.get_birefringence_mueller_matrix(delta[..., ii], alpha[..., ii])
            biref_obs = self.zodipol.imager.apply_birefringence(obs[ii], biref_mueller)
            I, Q, U = biref_obs.I, biref_obs.Q, biref_obs.U
            img_model = IQU_to_image(I, Q, U, cur_p, cur_eta)
            forward_results.append(img_model)
        results = np.stack(forward_results, axis=-1)
        return results

    def calibrate(self, n_itr=20, mode="all", disable=False, callback=None):
        self.initialize()
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
            # img_diff = img - img.mean(axis=(-2, -1), keepdims=True)
            # img_gt_diff = self.images.value - self.images.value.mean(axis=(-2, -1), keepdims=True)
            diff_resh = img - self.images.value
            cost = 1e20 * (diff_resh ** 2).sum(axis=(-2)) ** 0.5
            cost = np.nan_to_num(cost, nan=0)  # zero out nans
            return cost.flatten()

        jac_sparsity = eye(getattr(self, property_name).size)
        x0 = getattr(self, property_name).flatten()
        bounds = self.get_property_bounds(property_name)
        p_lsq = least_squares(cost_function, x0=x0.flatten(), jac_sparsity=jac_sparsity, ftol=1e-4, max_nfev=30,
                              bounds=bounds, verbose=2)
        return p_lsq.x.reshape(getattr(self, property_name).shape)

    @staticmethod
    def get_property_bounds(property_name):
        if property_name == "p":
            return 0.5, 1
        elif property_name == "eta":
            return -np.pi/4, np.pi/4
        elif property_name == "delta":
            return 0, np.pi
        elif property_name == "alpha":
            return -np.pi, np.pi
        else:
            raise ValueError("Unknown property name")

    def estimate_observations(self):
        rotation_list = self.rotation_list
        interp_images_res = self.aligned_images
        est_shape = (interp_images_res.shape[0], np.prod(interp_images_res.shape[1:]))
        p = self.align_images(self.p, rotation_list)[..., None, :].repeat(self.parser["n_polarization_ang"], axis=-2)
        eta = self.align_images(self.eta[:, None, :] + self.parser["polarization_angle"][:, None], rotation_list) + np.deg2rad(rotation_list)
        I_est, Q_est, U_est = estimate_IQU(interp_images_res.reshape(est_shape), p.reshape(est_shape), eta.reshape(est_shape))
        obs_est = Observation(I_est, Q_est, U_est, theta=self.theta, phi=self.phi, roll=0)
        obs_est_rot = [self.realign_observation(obs_est, roll) for roll in rotation_list]
        return obs_est_rot

    def align_images(self, images_res, rotation_arr, invert=False):
        if invert:
            rotation_arr = -rotation_arr
        res_images = []
        for ii in range(images_res.shape[-1]):
            rot_image = self.get_rotated_image(images_res[..., ii], rotation_arr[ii])
            res_images.append(rot_image)
        return np.stack(res_images, axis=-1)

    def realign_observation(self, obs, roll):
        I_interp = self.get_rotated_image(obs.I, -roll)
        Q_interp = self.get_rotated_image(obs.Q, -roll)
        U_interp = self.get_rotated_image(obs.U, -roll)
        obs = Observation(I_interp, Q_interp, U_interp, theta=obs.theta, phi=obs.phi, roll=0)
        return obs.change_roll(np.deg2rad(roll))

    def get_rotated_image(self, images, rotation_to):
        if rotation_to == 0:  # avoid interpolation issues
            return images
        theta_from, phi_from = self.zodipol._create_sky_coords(theta=self.parser["direction"][0], phi=self.parser["direction"][1],
                                                          roll=0 * u.deg, resolution=self.parser["resolution"])
        theta_to, phi_to = self.zodipol._create_sky_coords(theta=self.parser["direction"][0], phi=self.parser["direction"][1],
                                                      roll=-rotation_to * u.deg, resolution=self.parser["resolution"])
        interp = griddata(points=np.stack((theta_from, phi_from), axis=-1).value, values=images,
                          xi=np.stack((theta_to, phi_to), axis=-1).value, method='linear', fill_value=np.nan)
        return interp
    