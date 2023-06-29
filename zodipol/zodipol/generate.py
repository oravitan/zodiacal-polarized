import os
import pickle as pkl
import numpy as np
import astropy.units as u

from tqdm import tqdm
from typing import List
from skimage.util import random_noise

from zodipol.utils.argparser import ArgParser
from zodipol.zodipol.zodipol import Zodipol
from zodipol.zodipol.observation import Observation


DEFAULT_PATH = 'saved_models/self_calibration_obs.pkl'


def get_observations(n_rotations: int, zodipol: Zodipol, parser: ArgParser,  rotations_file_path: str = DEFAULT_PATH):
    """
    Generate observations for self-calibration.
    IF the saved rotations pickle file exists, then the observations are loaded from the file.
    ELSE, the observations are generated and saved to the file.
    """
    rotation_list = np.linspace(0, 360, n_rotations, endpoint=False)
    if os.path.isfile(rotations_file_path):
        # saved rotations pickle file exists
        with open(rotations_file_path, 'rb') as f:
            obs_rot = pkl.load(f)
        ind_list = np.linspace(0, len(obs_rot), n_rotations, endpoint=False, dtype=int)
        rotation_list = np.linspace(0, 360, len(obs_rot), endpoint=False)[ind_list]
        obs_rot = [obs_rot[ii] for ii in ind_list]
    else:
        obs_rot = [zodipol.create_observation(theta=parser["direction"][0], phi=parser["direction"][1], roll=t * u.deg,
                                              lonlat=False, new_isl=parser["new_isl"]) for t in tqdm(rotation_list)]
        with open(rotations_file_path, 'wb') as f:
            pkl.dump(obs_rot, f)
    return obs_rot, rotation_list


def get_initial_parameters(obs: List[Observation], parser: ArgParser, zodipol: Zodipol, mode: str = 'linear',
                           direction_uncertainty: u.Quantity = None):
    """
    Generate initial parameters for calibration.
    """
    # n_rotations = len(obs)
    # motion_blur = 360 / n_rotations * u.deg
    if direction_uncertainty is None:
        direction_uncertainty = parser["direction_uncertainty"]
    obs = [o.add_radial_blur(direction_uncertainty, list(parser["resolution"])) for o in obs]
    obs = [o.add_direction_uncertainty(parser["fov"], parser["resolution"], direction_uncertainty) for o in obs]
    obs = [o.dilate_star_pixels(3 + (direction_uncertainty / zodipol.fov * np.min(parser['resolution'])).value.astype(int), parser["resolution"]) for o in obs]

    if mode == 'linear':
        delta_val, phi_val = np.pi / 8, np.pi / 6
        delta = zodipol.imager.get_birefringence_mat(delta_val, 'center', flat=True)
        phi = zodipol.imager.get_birefringence_mat(phi_val, 'linear', flat=True, angle=np.pi / 2)
        mueller_truth = zodipol.imager.get_birefringence_mueller_matrix(delta, phi)
        obs_biref = [zodipol.imager.apply_birefringence(o, mueller_truth[:, None, ...]) for o in obs]

        polarizance = zodipol.imager.get_birefringence_mat(0.95, 'linear', flat=True, min=0.75)
        polarizance_real = polarizance.reshape((-1, 1, 1)).repeat(parser["n_polarization_ang"], axis=-1)
    elif mode == 'constant':
        delta_val, phi_val = np.pi / 8, np.pi / 6
        delta = zodipol.imager.get_birefringence_mat(delta_val, 'constant', flat=True)
        phi = zodipol.imager.get_birefringence_mat(phi_val, 'constant', flat=True, angle=np.pi / 2)
        mueller_truth = zodipol.imager.get_birefringence_mueller_matrix(delta, phi)
        obs_biref = [zodipol.imager.apply_birefringence(o, mueller_truth[:, None, ...]) for o in obs]

        polarizance = zodipol.imager.get_birefringence_mat(0.95, 'constant', flat=True)
        polarizance_real = polarizance.reshape((-1, 1, 1)).repeat(parser["n_polarization_ang"], axis=-1)
    elif mode == 'sine':
        delta_val, phi_val = np.pi / 8, 2 * np.pi
        delta = zodipol.imager.get_birefringence_mat(delta_val, 'center', flat=True, inv=True)
        phi = zodipol.imager.get_birefringence_mat(phi_val, 'sine', flat=True, angle=np.pi/2, min=-2*np.pi)
        mueller_truth = zodipol.imager.get_birefringence_mueller_matrix(delta, phi)

        obs_biref = [zodipol.imager.apply_birefringence(o, mueller_truth[:, None, ...]) for o in obs]

        polarizance = zodipol.imager.get_birefringence_mat(2 * np.pi, 'sine', flat=True, min=-2 * np.pi)
        polarizance = (polarizance - polarizance.min()) / (polarizance.max() - polarizance.min()) * 0.09 + 0.9
        polarizance_real = polarizance.reshape((-1, 1, 1)).repeat(parser["n_polarization_ang"], axis=-1)
    elif mode == 'anomalies':
        anomaly_amount, anomaly_percentage = 0.05, 0.03
        delta = zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)
        phi = zodipol.imager.get_birefringence_mat(0, 'constant', flat=True)
        mueller_truth = zodipol.imager.get_birefringence_mueller_matrix(delta, phi)
        obs_biref = [zodipol.imager.apply_birefringence(o, mueller_truth[:, None, ...]) for o in obs]

        polarizance = zodipol.imager.get_birefringence_mat(0.99, 'constant', flat=True)
        polarizance_noise = random_noise(polarizance, mode='pepper', amount=anomaly_percentage)
        polarizance = (1 - anomaly_amount) * polarizance + anomaly_amount * polarizance_noise  # reduce by anomaly_amount
        polarizance_real = polarizance.reshape((-1, 1, 1)).repeat(parser["n_polarization_ang"], axis=-1)
    else:
        raise ValueError(f'mode {mode} not recognized')
    polarization_angle = parser["polarization_angle"]
    polarization_angle_spatial_diff = np.zeros_like(polarizance)
    polarization_angle_real = polarization_angle[None, None, :] + polarization_angle_spatial_diff.flatten()[:, None, None]

    obs_orig = [zodipol.make_camera_images(o, polarizance_real, polarization_angle_real, n_realizations=parser["n_realizations"],
                                           add_noise=True) for o in obs_biref]
    images_orig = zodipol.post_process_images(np.stack(obs_orig, axis=-1))
    return obs, images_orig, polarizance_real.reshape(-1, parser["n_polarization_ang"]), polarization_angle_spatial_diff.reshape((-1)), mueller_truth


def get_initialization(p, mueller):
    p_noise = 0.02 + 0.01 * np.random.randn(p.shape[0])
    a, b, c = mueller[:, 1, 1], mueller[:, 1, 2], mueller[:, 2, 2]
    mueller_noise = 0.02 * np.random.randn(*a.shape + (3,))

    p_res = np.clip(p[:, 0] + p_noise, 0, 1)[..., None].repeat(4, axis=-1)
    a = np.clip(a.copy() + mueller_noise[..., 0], -1, 1)
    b = np.clip(b.copy() + mueller_noise[..., 1], -1, 1)
    c = np.clip(c.copy() + mueller_noise[..., 2], -1, 1)

    mueller_res = mueller.copy()[..., :3, :3]
    mueller_res[:, 1, 1] = a
    mueller_res[:, 1, 2] = b
    mueller_res[:, 2, 1] = b
    mueller_res[:, 2, 2] = c

    return {'p': p_res, 'biref': mueller_res}
