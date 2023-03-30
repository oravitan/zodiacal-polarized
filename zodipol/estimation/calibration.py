import numpy as np

from tqdm import tqdm


class Calibration:
    def __init__(self, obs, zodipol, parser):
        self.obs = obs
        self.zodipol = zodipol
        self.parser = parser

        self.p = None
        self.eta = None
        self.delta = None
        self.alpha = None

    def initialize(self):
        self.p = np.ones(self.parser["resolution"])
        self.eta = np.zeros(self.parser["resolution"])
        self.delta = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)
        self.alpha = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)

    def forward_model(self, o, p=None, eta=None, delta=None, alpha=None):
        p = (p if p is not None else self.p)
        eta = (eta if eta is not None else self.eta)
        delta = (delta if delta is not None else self.delta)
        alpha = (alpha if alpha is not None else self.alpha)

        biref_mueller = self.zodipol.imager.get_birefringence_mueller_matrix(delta, alpha)
        biref_obs = self.zodipol.imager.apply_birefringence(o, biref_mueller)
        img_model = self.zodipol.make_camera_images(biref_obs, p, eta, n_realizations=1, add_noise=False)
        return img_model

    def calibrate(self, images_orig, n_itr=10, mode="all", disable=False):
        self.initialize()
        itr_cost = []
        for ii in range(n_itr):
            self._calibrate_itr(images_orig, mode=mode, disable=disable)
            itr_cost.append(self.get_mse(images_orig))
        return self.p, self.eta, self.delta, self.alpha, itr_cost

    def get_mse(self, images_orig):
        img_model = self.forward_model(self.obs)
        mse = np.nanmean((img_model - images_orig) ** 2)
        return mse

    def _calibrate_itr(self, images_orig, mode="all", disable=False):
        if mode == "all":
            self.p = self._calibrate_property(images_orig, "p", 0.5, 1, 21, disable=disable)
            self.eta = self._calibrate_property(images_orig, "eta", -np.pi/8, np.pi/8, 21, disable=disable)
            self.delta = self._calibrate_property(images_orig, "delta", -np.pi/2, np.pi/2, 21, disable=disable)
            self.alpha = self._calibrate_property(images_orig, "alpha", -np.pi/2, np.pi/2, 21, disable=disable)
        elif mode == "P,eta":
            self.p = self._calibrate_property(images_orig, "p", 0.5, 1, 21, disable=disable)
            self.eta = self._calibrate_property(images_orig, "eta", np.deg2rad(-15), np.deg2rad(15), 21,disable=disable)
        elif mode == "delta,alpha":
            self.delta = self._calibrate_property(images_orig, "delta", -np.pi/2, np.pi/2, 21, disable=disable)
            self.alpha = self._calibrate_property(images_orig, "alpha", -np.pi/2, np.pi/2, 21, disable=disable)
        else:
            raise ValueError("Unknown mode")

    def _calibrate_property(self, images_orig, property_name, min_val, max_val, n_steps, disable=False):
        p_t = np.linspace(min_val, max_val, n_steps)
        img = [np.stack([self.forward_model(o, **{property_name: p}) for p in p_t], axis=-1) for o in tqdm(self.obs, disable=disable)]
        img_stack = np.stack(img, axis=-2)
        diff_resh = (img_stack - images_orig[..., None]).value.reshape(
            self.parser["resolution"] + [self.parser["n_polarization_ang"], len(self.obs), len(p_t)])

        c = np.nansum((1e23 * diff_resh) ** 2, axis=(-3, -2))
        p_est = p_t[np.argmin(np.nan_to_num(c, nan=np.inf), axis=-1)]
        return p_est

