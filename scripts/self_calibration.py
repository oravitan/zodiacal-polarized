import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.interpolate import griddata
from tqdm import tqdm
from itertools import repeat

# import the necessary modules
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol, Observation
from zodipol.estimation.estimate_signal import estimate_IQU
from zodipy.zodipy import IQU_to_image


def get_images(obs_est_rot, polarization_angle, polarizance, zodipol, parser, disable=False):
    img_list = []
    for ii in tqdm(range(len(obs_est_rot)), disable=disable):
        p, pa, roll = polarizance[..., ii], polarization_angle[..., ii], obs_est_rot[ii].roll
        # p = get_rotated_image(zodipol, p, parser, -np.rad2deg(roll))
        # pa = get_rotated_image(zodipol, pa, parser, -np.rad2deg(roll))
        I, Q, U = obs_est_rot[ii].I, obs_est_rot[ii].Q, obs_est_rot[ii].U
        cur_img = IQU_to_image(I, Q, U, p, pa + roll)
        img_list.append(cur_img)
    # roll_arr = np.array([o.roll for o in obs_est_rot])
    img_stack = np.stack(img_list, axis=-1)
    # img_align = align_images(zodipol, parser, img_stack, -np.rad2deg(roll_arr))
    return img_stack

def estimate_polarizance(obs_est_rot, images_orig, polarization_angle, zodipol, parser, disable=False):
    # perform grid search for the polarizance
    p_t = np.linspace(0.5, 1, 31)
    img_list = []
    for ii in tqdm(range(len(obs_est_rot)), disable=disable):
        pa, roll = polarization_angle[..., ii], obs_est_rot[ii].roll
        # pa = get_rotated_image(zodipol, pa, parser, -np.rad2deg(roll))
        I, Q, U = obs_est_rot[ii].I, obs_est_rot[ii].Q, obs_est_rot[ii].U
        cur_img = np.stack([IQU_to_image(I, Q, U, p, pa + roll) for p in p_t], axis=-1)
        img_list.append(cur_img)
    img_stack = np.stack(img_list, axis=-2)
    diff_resh = (img_stack - images_orig[..., None]).reshape(parser["resolution"] + list(img_stack.shape[1:]))

    c = np.nansum((1e23 * diff_resh) ** 2, axis=(-3))
    p_est = p_t[np.argmin(np.nan_to_num(c, nan=np.inf), axis=-1)]
    return p_est

def estimate_polarization_angle(obs_est_rot, images_orig, polarizance, zodipol, parser, disable=False):
    a_t = np.deg2rad(np.linspace(-10, 10, 31))
    base_polarization_ang = parser["polarization_angle"][None, ...]
    img_list = []
    for ii in tqdm(range(len(obs_est_rot)), disable=disable):
        p, roll = polarizance[..., ii], obs_est_rot[ii].roll
        # p = get_rotated_image(zodipol, p, parser, -np.rad2deg(roll))
        I, Q, U = obs_est_rot[ii].I, obs_est_rot[ii].Q, obs_est_rot[ii].U
        cur_img = np.stack([IQU_to_image(I, Q, U, p[..., None], base_polarization_ang + pa + roll) for pa in a_t], axis=-1)
        img_list.append(cur_img)
    img_stack = np.stack(img_list, axis=-2)
    diff_resh = (img_stack - images_orig[..., None]).reshape(parser["resolution"] + list(img_stack.shape[1:]))

    c = np.nansum((1e21 * diff_resh) ** 2, axis=(-3))
    p_est = a_t[np.argmin(np.nan_to_num(c, nan=np.inf), axis=-1)]
    return p_est

def estimate_observations(zodipol, parser, rotation_list, images, polarizance, angles, theta, phi):
    est_shape = (images.shape[0], np.prod(images.shape[1:]))
    I_est, Q_est, U_est = estimate_IQU(images.reshape(est_shape), polarizance.reshape(est_shape), angles.reshape(est_shape))
    obs_est = Observation(I_est, Q_est, U_est, theta=theta, phi=phi, roll=0)
    # obs_est_rot = [obs_est.change_roll(roll) for roll in np.deg2rad(roll_list)]
    # obs_est_rot = [obs_est] * len(roll_list)
    obs_est_rot = list(map(realign_observation, repeat(zodipol), repeat(parser), repeat(obs_est), rotation_list))
    return obs_est_rot

def get_rotated_image(zodipol, images, parser, rotation_to):
    if rotation_to == 0:  # avoid interpolation issues
        return images
    theta_from, phi_from = zodipol._create_sky_coords(theta=parser["direction"][0], phi=parser["direction"][1], roll=0 * u.deg, resolution=parser["resolution"])
    theta_to, phi_to = zodipol._create_sky_coords(theta=parser["direction"][0], phi=parser["direction"][1], roll=-rotation_to * u.deg, resolution=parser["resolution"])
    interp = griddata(points=np.stack((theta_from, phi_from), axis=-1).value, values=images, xi=np.stack((theta_to, phi_to), axis=-1).value, method='linear', fill_value=np.nan)
    return interp

def align_images(zodipol, parser, images_res, rotation_arr, invert=False):
    if invert:
        rotation_arr = -rotation_arr
    res_images = []
    for ii in range(images_res.shape[-1]):
        rot_image = get_rotated_image(zodipol, images_res[..., ii], parser, rotation_arr[ii])
        res_images.append(rot_image)
    return np.stack(res_images, axis=-1)

def realign_observation(zodipol, parser, obs, roll):
    I_interp = get_rotated_image(zodipol, obs.I, parser, -roll)
    Q_interp = get_rotated_image(zodipol, obs.Q, parser, -roll)
    U_interp = get_rotated_image(zodipol, obs.U, parser, -roll)
    return Observation(I_interp, Q_interp, U_interp, theta=obs.theta, phi=obs.phi, roll=np.deg2rad(roll))


def get_measurements(zodipol, parser, n_rotations=40):
    rotations_file_path = 'saved_models/self_calibration_40.pkl'
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


def calculate_errors(est, truth):
    error = np.nanmean((est - truth) ** 2, axis=0)
    mean_err = np.nanmean(error)
    estimation_std = np.nanstd(error)
    return mean_err, estimation_std


def generate_observations(zodipol, parser, n_rotations=40):
    obs_rot, rotation_list = get_measurements(zodipol, parser, n_rotations=n_rotations)

    # create satellite polarizance and angle of polarization variables
    polarization_angle = parser["polarization_angle"]
    _, polarization_angle_spatial_diff = np.meshgrid(np.arange(parser["resolution"][0]),
                                                     np.deg2rad(np.linspace(-10, 10, parser["resolution"][1])),
                                                     indexing='ij')

    pa_ts_diff = polarization_angle_spatial_diff.flatten()[:, None, None][..., None] * np.linspace(0, 1, n_rotations)
    polarization_angle_real = polarization_angle[None, None, :, None] + pa_ts_diff

    polarizance, _ = np.meshgrid(np.linspace(-1, 0, parser["resolution"][0]), np.arange(parser["resolution"][1]),
                                 indexing='ij')
    polarizance_real = polarizance.reshape((len(obs_rot[0]), 1, 1))
    polarizance_real = 1 + polarizance_real[..., None] * np.linspace(0, 0.4, n_rotations)

    # create observations images
    obs_orig = [zodipol.make_camera_images(obs_rot[ii], polarizance_real[..., ii], polarization_angle_real[..., ii],
                                                 n_realizations=parser["n_realizations"], add_noise=True) for ii in range(n_rotations)]
    images_orig = np.stack(obs_orig, axis=-1)
    images_res = images_orig.reshape((parser["resolution"] + list(images_orig.shape[1:])))
    return images_res, rotation_list, polarizance_real, polarization_angle_real


def perform_estimation(zodipol, parser, rotation_list, images_res_flat, polarizance_real, polarization_angle_real,
                       disable=False):
    n_pixels = np.prod(parser["resolution"])
    polarization_angle = parser["polarization_angle"]
    n_rotations = len(rotation_list)

    theta0, phi0 = zodipol._create_sky_coords(theta=parser["direction"][0], phi=parser["direction"][1], roll=0 * u.deg, resolution=parser["resolution"])
    interp_images_res = align_images(zodipol, parser, images_res_flat.value, rotation_list)
    nan_ind = np.isnan(interp_images_res).any(axis=-1, keepdims=True)

    init_polarizance, init_polarization_ang = np.ones(parser["resolution"]), np.zeros(parser["resolution"])
    polarizance_est = np.ones((n_pixels, n_rotations))
    polarization_ang_est = np.zeros((n_pixels, n_rotations))

    polarization_ang_full = polarization_ang_est[..., None, :] + polarization_angle[:, None]
    polarization_ang_align = align_images(zodipol, parser, polarization_ang_full, rotation_list)
    polarization_ang_rot = polarization_ang_align + np.deg2rad(rotation_list[None, None, :])

    polarizance_est_reshape = np.broadcast_to(polarizance_est[..., None, :], polarization_ang_rot.shape)
    polarizance_est_align = align_images(zodipol, parser, polarizance_est_reshape, rotation_list)

    # organize inputs for estimation
    cost_itr = {'total': [], 'polarizance': [], 'polarization_ang': []}
    for ii in tqdm(range(6), disable=disable):
        # estimate observations
        obs_est_rot = estimate_observations(zodipol, parser, rotation_list, interp_images_res,
                                            polarizance_est_align, polarization_ang_rot, theta0, phi0)

        # estimate polarizance and angle of polarization
        est_pol = estimate_polarizance(obs_est_rot[1:], images_res_flat.value[..., 1:], polarization_ang_full[..., 1:],
                                       zodipol, parser, disable=True)
        est_pol = np.concatenate((init_polarizance[..., None], est_pol), axis=-1)
        polarizance_est_reshape = np.broadcast_to(est_pol.reshape((n_pixels, 1, n_rotations)),
                                                  polarization_ang_rot.shape)
        polarizance_est_align = align_images(zodipol, parser, polarizance_est_reshape, rotation_list)

        est_pa = estimate_polarization_angle(obs_est_rot[1:], images_res_flat.value[..., 1:], polarizance_est[..., 1:],
                                             zodipol, parser, disable=True)
        est_pa = np.concatenate((init_polarization_ang[..., None], est_pa), axis=-1)
        polarization_ang_est = est_pa.reshape((n_pixels, n_rotations))
        polarization_ang_full = polarization_ang_est[..., None, :] + polarization_angle[:, None]
        polarization_ang_align = align_images(zodipol, parser, polarization_ang_full, rotation_list)
        polarization_ang_rot = polarization_ang_align + np.deg2rad(rotation_list[None, None, :])

        est_images = get_images(obs_est_rot, polarization_ang_full, polarizance_est_reshape, zodipol, parser,
                                disable=True)
        cost = np.sqrt(np.nansum(1e23 * (images_res_flat.value - est_images) ** 2))
        polarizance_cost = np.sqrt(np.sum((polarizance_real.squeeze() - polarizance_est_reshape[:, 0, :]) ** 2))
        polarization_ang_cost = np.sqrt(np.sum((polarization_angle_real.squeeze() - polarization_ang_full) ** 2))
        cost_itr['total'].append(cost)
        cost_itr['polarizance'].append(polarizance_cost)
        cost_itr['polarization_ang'].append(polarization_ang_cost)
    polarizance_est_reshape = np.where(nan_ind, np.nan, polarizance_est_reshape)
    polarization_ang_full = np.where(nan_ind, np.nan, polarization_ang_full)
    return cost_itr, polarizance_est_reshape, polarization_ang_full


def main():
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    # generate observations
    polariz_bias_l, polariz_std_l, pol_ang_bias_l, pol_ang_std_l = [], [], [], []
    n_rotation_list = np.array([5, 9, 13, 17, 21])
    for n_rotations in n_rotation_list:
        images_res, rotation_list, polarizance_real, polarization_angle_real = generate_observations(zodipol, parser, n_rotations=n_rotations)
        images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
        cost_itr, polarizance_est_reshape, polarization_ang_full = perform_estimation(zodipol, parser, rotation_list, images_res_flat, polarizance_real, polarization_angle_real)
        polariz_bias, polariz_std = calculate_errors(polarizance_est_reshape[:, 0, 1:], polarizance_real.squeeze()[..., 1:])
        pol_ang_bias, pol_ang_std = calculate_errors(polarization_ang_full[..., 1:], polarization_angle_real.squeeze()[..., 1:])
        polariz_bias_l.append(polariz_bias)
        polariz_std_l.append(polariz_std)
        pol_ang_bias_l.append(pol_ang_bias)
        pol_ang_std_l.append(pol_ang_std)

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(n_rotation_list, y=polariz_bias_l, yerr=polariz_std_l, capsize=10)
    ax2 = ax.twinx()
    ax2.errorbar(0.1+n_rotation_list, y=pol_ang_bias_l, yerr=pol_ang_std_l, color='r', capsize=10)
    ax.set_xlabel('Number of Rotations', fontsize=14)
    ax.set_ylabel('Polarizance Bias', fontsize=14)
    ax2.set_ylabel('Polarization Angle Bias', fontsize=14)
    plt.title('Bias and Standard Deviation \nof Polarizance and Polarization Angle', fontsize=16)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()