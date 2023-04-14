import numpy as np

from tqdm import tqdm
from scipy.optimize import least_squares
from scipy.sparse import eye
from scipy.signal import convolve2d

from zodipol.zodipol.zodipol import IQU_to_image


class Calibration:
    def __init__(self, obs, zodipol, parser):
        self.obs = obs
        self.zodipol = zodipol
        self.parser = parser
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

    def forward_model(self, o, p=None, eta=None, biref=None):
        p = (p if p is not None else self.p)
        eta = (eta if eta is not None else self.eta)
        biref = (biref if biref is not None else self.biref)

        p = p.reshape((-1, 1))
        eta = eta.reshape((-1, 1)) + self.parser["polarization_angle"][None, :]

        biref_obs = self.zodipol.imager.apply_birefringence(o, biref)
        img_model = IQU_to_image(biref_obs.I, biref_obs.Q, biref_obs.U, p, eta)
        return img_model

    def calibrate(self, images_orig, n_itr=5, mode="all", disable=False, callback=None, init=None):
        self.initialize(init)
        itr_cost = [self.get_mse(images_orig)]
        itr_callback = []
        if callback is not None:
            itr_callback.append(callback(self))
        for _ in tqdm(range(n_itr), disable=disable):
            self._calibrate_itr(images_orig, mode=mode)
            itr_cost.append(self.get_mse(images_orig))
            if callback is not None:
                itr_callback.append(callback(self))
        if callback is not None:
            return self.p, self.eta, self.biref, itr_cost, itr_callback
        return self.p, self.eta, self.biref, itr_cost

    def get_mse(self, images_orig):
        img_model = np.stack([self.forward_model(o) for o in self.obs], axis=-1)
        mse = np.nanmean((img_model - images_orig) ** 2)
        A_gamma = self.zodipol.imager.get_A_gamma(self.zodipol.frequency, self.zodipol.get_imager_response())
        mse_electrons = (np.sqrt(mse) / A_gamma).si.value.squeeze() ** 2
        return mse_electrons

    def _calibrate_itr(self, images_orig, mode="all"):
        self.estimate_p_eta(images_orig)
        self.estimate_delta_eta(images_orig)

    def estimate_p_eta(self, images_orig):
        # preparation
        intensity = images_orig.value
        mueller = self.biref
        stokes = np.stack([o.to_numpy(ndims=3) for o in self.obs], axis=-2).value

        # M_eta = 0.5 * np.stack((np.ones_like(angles), np.cos(2 * angles), np.sin(2 * angles)), axis=-1)
        # A = np.einsum('...ai, ...ij,...jk->...iak', M_eta, mueller, stokes)

        F_P = np.einsum('...ij,...jk->...ik', stokes, mueller)
        pseudo_inv = np.linalg.pinv(F_P)
        M_p_eta_inv = np.einsum('...ij,...kj->...ik', pseudo_inv, intensity)
        p_est = (M_p_eta_inv[:, 1, 0] + M_p_eta_inv[:, 2, 1] - M_p_eta_inv[:, 1, 2] - M_p_eta_inv[:, 2, 3])/2
        self.p = np.clip(p_est, 0, 1)

    def estimate_delta_eta(self, images_orig, kernel_size=5):
        intensity = images_orig.value
        p = self.p[:, None]
        angles = self.eta[:, None] + self.parser["polarization_angle"]

        # preparation
        stokes = np.stack([o.to_numpy(ndims=3) for o in self.obs], axis=-1).value
        angles_matrix = 0.5 * np.stack((np.ones_like(angles), p * np.cos(2 * angles), p * np.sin(2 * angles)), axis=-1)

        stokes_pseudo_inv = np.linalg.pinv(stokes)
        angles_pseudo_inv = np.linalg.pinv(angles_matrix)

        biref = np.einsum('...ij,...jk,...kw->...iw', angles_pseudo_inv, intensity, stokes_pseudo_inv)

        # smooth biref
        kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
        biref_resh = biref.reshape(self.parser["resolution"] + [9])
        biref_smooth = np.stack([convolve2d(biref_resh[..., ii], kernel, mode='same', boundary='symm') for ii in range(biref_resh.shape[-1]) ], axis=-1)
        biref = biref_smooth.reshape(biref.shape)

        # set necessary values of biref
        # biref_fixed = np.clip(biref, -1, 1).reshape(biref.shape)
        biref[:, 0, 0] = 1  # force to avoid numerical errors
        biref[:, 0, 1] = biref[:, 1, 0] = biref[:, 0, 2] = biref[:, 2, 0] = 0
        self.biref = biref
