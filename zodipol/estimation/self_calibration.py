import numpy as np
import astropy.units as u

from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
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
        self.aligned_images = self.align_images(images_res_flat.value, rotation_list, fill_value=0)
        self.nan_mask = self.get_nan_mask()
        self.initialize()

    def initialize(self, init = None):
        self.p = np.ones(self.parser["resolution"]).reshape((-1))
        self.eta = np.zeros(self.parser["resolution"]).reshape((-1))
        delta = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)
        alpha = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)
        if init is not None and 'p' in init:
            self.p = init['p']
        if init is not None and 'eta' in init:
            self.eta = init['eta']
        if init is not None and 'delta' in init:
            delta = init['delta']
        if init is not None and 'alpha' in init:
            alpha = init['alpha']
        self.biref = self.zodipol.imager.get_birefringence_mueller_matrix(delta, alpha)[..., :3, :3]

    def get_nan_mask(self):
        nan_imag = self.align_images(np.ones((self.images.shape[0], len(self.rotation_list))), self.rotation_list, fill_value=np.nan)
        nan_ind = np.isnan(nan_imag).any(axis=-1).squeeze()
        return nan_ind

    def get_properties(self):
        p_nan = np.where(self.nan_mask, np.nan, self.p)
        eta_nan = np.where(self.nan_mask, np.nan, self.eta)
        biref = self.biref.copy()
        biref[self.nan_mask, ...] = np.nan
        return p_nan, eta_nan, biref

    def forward_model(self, obs=None, p=None, eta=None, biref=None):
        obs = (obs if obs is not None else self.obs)
        p = (p if p is not None else self.p)
        eta = (eta if eta is not None else self.eta)
        biref = (biref if biref is not None else self.biref)

        forward_results = []
        for ii in range(len(obs)):
            cur_p = p[:, None]
            cur_eta = eta[:, None] + self.parser["polarization_angle"]
            biref_obs = self.zodipol.imager.apply_birefringence(obs[ii], biref)

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
        self.estimate_p_eta()
        self.estimate_delta_eta()

    def estimate_observations(self):
        rotation_list = self.rotation_list
        # interp_images_res = self.zodipol.post_process_images(self.aligned_images, sign=-1)
        interp_images_res = self.aligned_images
        # est_shape = (interp_images_res.shape[0], np.prod(interp_images_res.shape[1:]))
        p = self.align_images(self.p[:, None].repeat(self.nobs, axis=-1), rotation_list, invert=True)[..., None, :].repeat(self.parser["n_polarization_ang"], axis=-2)
        eta = self.align_images(self.eta[:, None, None].repeat(self.nobs, axis=-1), rotation_list, invert=True) + self.parser["polarization_angle"][:, None]
        mueller = self.biref
        mueller_rot = mueller[..., None].repeat(self.nobs, axis=-1)
        rotation_mat = get_rotation_mueller_matrix(np.deg2rad(rotation_list))[None, ...].repeat(self.images.shape[0], axis=0)
        I_est, Q_est, U_est = self.estimate_IQU(interp_images_res, rotation_mat, p, eta, mueller_rot)
        obs_est = Observation(I_est, Q_est, U_est, theta=self.theta, phi=self.phi, roll=0)
        obs_est_rot = [self.realign_observation(obs_est, roll) for roll in rotation_list]
        return obs_est_rot

    def estimate_IQU(self, intensity, rotation_mat, polarizance, angles, mueller):
        """
        Calculate the Stokes parameters of the signal.
        :param intensity: The intensity of the signal.
        :param angles: The angles of the signal.
        :return: The Stokes parameters of the signal.
        """
        # create the angles matrix
        angles_matrix = 0.5 * np.stack((np.ones_like(angles), polarizance*np.cos(2*angles), polarizance*np.sin(2*angles)), axis=-3)
        forward_mat = np.einsum('...ijk,...iwk,...kws->...sjk', angles_matrix, mueller[:, :3, :3, :], rotation_mat[..., :3, :3])
        angles_matrix[self.nan_mask, ...] = forward_mat[self.nan_mask, ...] = np.nan

        pseudo_inverse = np.einsum('...ijk,...wjk->...iw', forward_mat, forward_mat)
        intensity_mult = np.einsum('...ijk,...jk->...i', forward_mat, intensity)
        intensity_inv = np.linalg.solve(pseudo_inverse, intensity_mult)
        I, Q, U = intensity_inv[..., 0], intensity_inv[..., 1], intensity_inv[..., 2]
        return I, Q, U

    def estimate_p_eta(self):
        # preparation
        angles = self.eta[:, None] + self.parser["polarization_angle"]
        intensity = self.images.value
        mueller = self.biref
        stokes = np.stack([o.to_numpy(ndims=3) for o in self.obs], axis=-1)

        M_eta = 0.5 * np.stack((np.ones_like(angles), np.cos(2 * angles), np.sin(2 * angles)), axis=-1)
        intensity[self.nan_mask, ...] = mueller[self.nan_mask, ...] = stokes[self.nan_mask, ...] = M_eta[self.nan_mask, ...] = np.nan

        A = np.einsum('...ai, ...ij,...jk->...iak', M_eta, mueller, stokes)

        # calculate the pseudo inverse
        A_A_T = np.einsum('...iak,...jak->...ij', A, A)
        A_I_T = np.einsum('...ijk,...jk->...i', A, intensity)

        M_p_eta_inv = np.linalg.solve(A_A_T, A_I_T)
        p = M_p_eta_inv[:, 1:].mean(axis=1)
        self.p = p - np.nanmean(p)  # p is solved up to a global shift - set max to 1

    def estimate_delta_eta(self, kernel_size = 5):
        intensity = self.images.value
        p = self.p[:, None]
        angles = self.eta[:, None] + self.parser["polarization_angle"]

        # preparation
        stokes = np.stack([o.to_numpy(ndims=3) for o in self.obs], axis=-1)
        angles_matrix = 0.5 * np.stack((np.ones_like(angles), p * np.cos(2 * angles), p * np.sin(2 * angles)), axis=-1)
        intensity[self.nan_mask, ...] = stokes[self.nan_mask, ...] = angles_matrix[self.nan_mask, ...] = np.nan

        # step 1
        pseudo_inverse_stokes = np.einsum('...ij,...kj->...ik', stokes, stokes)[:, None, ...]
        st_inv = np.einsum('...ij,...kj->...ki', stokes, intensity)
        M_p_M_biref = np.linalg.solve(pseudo_inverse_stokes, st_inv)

        # step 2
        pseudo_inverse_angles = np.einsum('...ki,...kj->...ij', angles_matrix, angles_matrix)
        angles_intensity = np.einsum('...ij,...ik->...kj', angles_matrix, M_p_M_biref)
        biref = np.linalg.solve(pseudo_inverse_angles, angles_intensity)

        # normalize biref
        W, V = np.linalg.eig(np.nan_to_num(biref[..., 1:, 1:], nan=0))
        biref[..., 1:, 1:] = biref[..., 1:, 1:] / np.max(W.real, axis=-1)[:, None, None]

        # smooth biref
        # kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
        # biref_resh = biref.reshape(self.parser["resolution"] + [9])
        # biref_smooth = np.stack([convolve2d(biref_resh[..., ii], kernel, mode='same') for ii in range(biref_resh.shape[-1]) ], axis=-1)

        # set necessary values of biref
        biref_fixed = np.clip(biref, None, 1).reshape(biref.shape)
        # biref_smooth = np.clip(biref_smooth, None, 1).reshape(biref.shape)
        biref_fixed[:, 0, 0] = 1  # force to avoid numerical errors
        biref_fixed[:, 0, 1] = biref_fixed[:, 1, 0] = biref_fixed[:, 0, 2] = biref_fixed[:, 2, 0] = 0
        self.biref = biref_fixed

    def align_images(self, images_res, rotation_arr, invert=False, fill_value=0, nan_edge=False):
        if invert:
            rotation_arr = rotation_arr
        res_images = []
        for ii in range(len(rotation_arr)):
            rot_image = self.get_rotated_image(images_res[..., ii], -rotation_arr[ii], fill_value=fill_value)
            res_images.append(rot_image)
        images_res = np.stack(res_images, axis=-1)
        return images_res

    def realign_observation(self, obs, roll):
        I_interp = self.get_rotated_image(obs.I, roll)
        Q_interp = self.get_rotated_image(obs.Q, roll)
        U_interp = self.get_rotated_image(obs.U, roll)
        obs_new = Observation(I_interp, Q_interp, U_interp, theta=obs.theta, phi=obs.phi, roll=0)
        return obs_new.change_roll(np.deg2rad(roll))

    def get_rotated_image(self, images, rotation_to, fill_value=0):
        images = np.nan_to_num(images, nan=fill_value)  # fill nans
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
    