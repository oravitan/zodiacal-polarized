import numpy as np

from tqdm import tqdm
from scipy.optimize import least_squares
from scipy.sparse import eye


class Calibration:
    def __init__(self, obs, zodipol, parser):
        self.obs = obs
        self.zodipol = zodipol
        self.parser = parser
        self.initialize()

    def initialize(self):
        self.p = np.ones(self.parser["resolution"]).reshape((-1))
        self.eta = np.zeros(self.parser["resolution"]).reshape((-1))
        self.delta = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)
        self.alpha = self.zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)

    def forward_model(self, o, p=None, eta=None, delta=None, alpha=None):
        p = (p if p is not None else self.p)
        eta = (eta if eta is not None else self.eta)
        delta = (delta if delta is not None else self.delta)
        alpha = (alpha if alpha is not None else self.alpha)

        p = p.reshape((-1, 1, 1))
        eta = eta.reshape((-1, 1, 1)) + self.parser["polarization_angle"][None, None, :]
        # delta = delta.squeeze()
        # alpha = alpha.squeeze()

        biref_mueller = self.zodipol.imager.get_birefringence_mueller_matrix(delta, alpha)
        biref_obs = self.zodipol.imager.apply_birefringence(o, biref_mueller)
        img_model = self.zodipol.make_camera_images(biref_obs, p, eta, n_realizations=1, add_noise=False)
        img_model = self.zodipol.post_process_images(img_model)
        return img_model

    def calibrate(self, images_orig, n_itr=5, mode="all", disable=False, callback=None):
        self.initialize()
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
            return self.p, self.eta, self.delta, self.alpha, itr_cost, itr_callback
        return self.p, self.eta, self.delta, self.alpha, itr_cost

    def get_mse(self, images_orig):
        img_model = np.stack([self.forward_model(o) for o in self.obs], axis=-1)
        mse = np.nanmean((img_model - images_orig) ** 2).value
        return mse

    def _calibrate_itr(self, images_orig, mode="all"):
        if mode == "all":
            self.p = self._calibrate_property(images_orig, "p")
            self.eta = self._calibrate_property(images_orig, "eta")
            self.delta = self._calibrate_property(images_orig, "delta")
            self.alpha = self._calibrate_property(images_orig, "alpha")
        elif mode == "P,eta":
            self.p = self._calibrate_property(images_orig, "p")
            self.eta = self._calibrate_property(images_orig, "eta")
        elif mode == "delta,alpha":
            self.delta = self._calibrate_property(images_orig, "delta")
            self.alpha = self._calibrate_property(images_orig, "alpha")
        else:
            raise ValueError("Unknown mode")

    def _calibrate_property(self, images_orig, property_name):
        def cost_function(x):
            img = [self.forward_model(o, **{property_name: x}) for o in self.obs]
            img_stack = np.stack(img, axis=-1)
            diff_resh = (img_stack - images_orig).value
            cost = 1e20*(diff_resh**2).sum(axis=(-2, -1))**0.5
            return cost

        jac_sparsity = eye(len(self.p))
        bounds = self.get_property_bounds(property_name)
        x0 = getattr(self, property_name).squeeze()
        p_lsq = least_squares(cost_function, x0=x0, jac_sparsity=jac_sparsity, ftol=1e-5, max_nfev=30, verbose=2, bounds=bounds)
        return p_lsq.x.reshape((-1))

    @staticmethod
    def get_property_bounds(property_name):
        if property_name == "p":
            return 0.5, 1
        elif property_name == "eta":
            return -np.pi/4, np.pi/4
        elif property_name == "delta":
            return -np.pi/2, np.pi/2
        elif property_name == "alpha":
            return -np.pi/2, np.pi/2
        else:
            raise ValueError("Unknown property name")
