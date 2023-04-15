import numpy as np
import astropy.units as u

from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator
from scipy.signal import convolve2d
from tqdm import tqdm

from zodipol.zodipol import Observation
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

    def initialize(self, init=None):
        if init is None:
            init = {}
        self.p = np.ones(self.parser["resolution"]).reshape((-1))
        self.eta = np.zeros(self.parser["resolution"]).reshape((-1))
        delta = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)
        alpha = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)
        if 'p' in init:
            self.p = init['p']
        if 'eta' in init:
            self.eta = init['eta']
        if 'delta' in init:
            delta = init['delta']
        if 'alpha' in init:
            alpha = init['alpha']
        self.biref = self.zodipol.imager.get_birefringence_mueller_matrix(delta, alpha)[..., :3, :3]

    def get_nan_mask(self):
        nan_imag = self.align_images(np.ones((self.images.shape[0], len(self.rotation_list))), self.rotation_list, fill_value=np.nan, method='linear')
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
        err = img_model*self.images.unit - self.images
        mse = np.nanmean(err[~self.nan_mask] ** 2)
        A_gamma = self.zodipol.imager.get_A_gamma(self.zodipol.frequency, self.zodipol.get_imager_response())
        mse_electrons = (np.sqrt(mse) / A_gamma).to('').value.squeeze() ** 2
        return mse_electrons

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
        mueller_rot = self.align_images(mueller_rot, rotation_list, invert=True)
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
        # intensity[self.nan_mask] = np.nan
        angles_matrix = 0.5 * np.stack((np.ones_like(angles), polarizance*np.cos(2*angles), polarizance*np.sin(2*angles)), axis=-3)
        # forward_mat = np.einsum('...ijk,...iwk,...kws->...jsk', angles_matrix, mueller[:, :3, :3, :], rotation_mat[..., :3, :3])
        forward_mat = np.einsum('...ijk,...iwk,...kws->...sjk', angles_matrix, mueller[:, :3, :3, :], rotation_mat[..., :3, :3])
        # angles_matrix[self.nan_mask, ...] = forward_mat[self.nan_mask, ...] = np.nan

        forward_resh = forward_mat.reshape(forward_mat.shape[:2] + (np.prod(forward_mat.shape[2:]),))
        intensity_resh = intensity.reshape((intensity.shape[0], np.prod(intensity.shape[1:])))

        pseudo_inverse = np.linalg.pinv(forward_resh)
        stokes_inv = np.einsum('...ij,...i->...j', pseudo_inverse, intensity_resh)
        I, Q, U = stokes_inv[..., 0], stokes_inv[..., 1], stokes_inv[..., 2]
        return I, Q, U

    def estimate_p_eta(self):
        # preparation
        intensity = self.images.value
        # intensity[self.nan_mask] = np.nan
        mueller = self.biref
        stokes = np.stack([o.to_numpy(ndims=3) for o in self.obs], axis=-2)

        F_P = np.einsum('...ij,...jk->...ik', stokes, mueller)
        # fp_filled = self.fill_outliers(F_P, np.isnan(F_P))
        pseudo_inv = np.linalg.pinv(F_P)
        M_p_eta_inv = np.einsum('...ij,...kj->...ik', pseudo_inv, intensity)
        p_est = (M_p_eta_inv[:, 1, 0] + M_p_eta_inv[:, 2, 1] - M_p_eta_inv[:, 1, 2] - M_p_eta_inv[:, 2, 3]) / 2


        p_est = p_est - np.nanmax(p_est[~self.nan_mask]) + 1
        p_est = np.clip(p_est, 0, 1)
        self.p = p_est

    def estimate_delta_eta(self, kernel_size=5):
        # intensity = self.aligned_images
        # intensity[self.nan_mask] = np.nan
        intensity = self.images.value.swapaxes(-1, -2)
        p = self.p[:, None]
        angles = self.eta[:, None] + self.parser["polarization_angle"]

        # preparation
        stokes = np.stack([o.to_numpy(ndims=3) for o in self.obs], axis=-1)
        angles_matrix = 0.5 * np.stack((np.ones_like(angles), p * np.cos(2 * angles), p * np.sin(2 * angles)), axis=-1)

        stokes_pseudo_inv = np.linalg.pinv(stokes.swapaxes(-1,-2))
        angles_pseudo_inv = np.linalg.pinv(angles_matrix.swapaxes(-1,-2))

        biref = np.einsum('...ij,...jk,...kw->...iw', stokes_pseudo_inv, intensity, angles_pseudo_inv)

        assert np.sqrt(np.median((biref[:, 0, 0] - 1)**2)) < 0.05, 'biref[0, 0] is not 1'

        # normalize biref
        W, V = np.linalg.eig(biref[:, 1:, 1:])
        eig_normalization = np.max((np.ones(biref.shape[:1]), np.max(W.real, axis=-1)),axis=0)[:, None, None]
        biref[..., 1:, 1:] = biref[..., 1:, 1:] / eig_normalization
        # W, V = np.linalg.eig(np.nan_to_num(biref[..., 1:, 1:], np.nanmean(biref[..., 1:, 1:])))
        # biref[..., 1:, 1:] = biref[..., 1:, 1:] / np.max((np.ones(biref.shape[:1]), np.max(abs(W.real), axis=-1)), axis=0)[:, None, None]

        # smooth biref
        # kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
        # biref_resh = biref.reshape(self.parser["resolution"] + [9])
        # biref_smooth = np.stack([convolve2d(biref_resh[..., ii], kernel, mode='same', boundary='symm') for ii in
        #                          range(biref_resh.shape[-1])], axis=-1)
        # biref = biref_smooth.reshape(biref.shape)

        # set necessary values of biref
        # biref_fixed = np.clip(biref, -1, 1).reshape(biref.shape)
        biref[:, 0, 0] = 1  # force to avoid numerical errors
        biref[:, 0, 1] = biref[:, 1, 0] = biref[:, 0, 2] = biref[:, 2, 0] = 0
        self.biref = biref

    def align_images(self, images_res, rotation_arr, invert=False, fill_value=0, method="nearest"):
        if invert:
            rotation_arr = rotation_arr
        res_images = []
        for ii in range(len(rotation_arr)):
            rot_image = self.get_rotated_image(images_res[..., ii], -rotation_arr[ii], fill_value=fill_value, method=method)
            res_images.append(rot_image)
        images_res = np.stack(res_images, axis=-1)
        return images_res

    def realign_observation(self, obs, roll):
        stokes_interp = self.get_rotated_image(obs.to_numpy(ndims=3), roll)
        # solve boundary issues by removing outliers
        obs_new = Observation(stokes_interp[..., 0], stokes_interp[..., 1], stokes_interp[..., 2], theta=obs.theta, phi=obs.phi, roll=0)
        return obs_new.change_roll(np.deg2rad(roll))


    # def fill_outliers(self, data, mask):
    #     non_mask_ind, mask_ind = np.nonzero(~mask), np.nonzero(mask)
    #     interp = NearestNDInterpolator(non_mask_ind, data[~mask])
    #     data[mask] = interp(mask_ind)
    #     return data


    def get_rotated_image(self, images, rotation_to, fill_value=0, method="nearest"):
        images = np.nan_to_num(images, nan=fill_value)  # fill nans
        if rotation_to == 0:  # avoid interpolation issues
            return images
        theta_from, phi_from = self.zodipol.create_sky_coords(theta=self.parser["direction"][0],
                                                              phi=self.parser["direction"][1],
                                                              roll=0 * u.deg, resolution=self.parser["resolution"])
        vec_from = ang2vec(theta_from, phi_from)
        x = np.linspace(vec_from[:, 0].min(), vec_from[:, 0].max(), self.parser["resolution"][1])
        y = np.linspace(vec_from[:, 1].min(), vec_from[:, 1].max(), self.parser["resolution"][0])
        images_resh = images.reshape(self.parser["resolution"] + list(images.shape[1:]))
        grid_interp = RegularGridInterpolator((y, x), images_resh, bounds_error=False, fill_value=fill_value, method=method)

        theta_to, phi_to = self.zodipol.create_sky_coords(theta=self.parser["direction"][0], phi=self.parser["direction"][1],
                                                          roll=rotation_to * u.deg, resolution=self.parser["resolution"])
        vec_to = ang2vec(theta_to, phi_to)
        interp = grid_interp(list(zip(vec_to[:, 1], vec_to[:, 0])))
        return interp
    