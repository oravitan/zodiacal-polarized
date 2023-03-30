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
from zodipol.zodipy_local.zodipy.zodipy import IQU_to_image


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


def get_iteration_cost(polarizance_real, polarizance_est_reshape_nan, polarization_angle_real, polarization_ang_full_nan):
    p_pdiff = polarizance_real.squeeze()[:, 0, None] - polarizance_real.squeeze()
    p_estdiff = polarizance_est_reshape_nan[:, 0, 0, None] - polarizance_est_reshape_nan[:, 0, :]

    eta_pdiff = polarization_angle_real.squeeze()[:, 0, 0, None] - polarization_angle_real.squeeze()[:, 0, :]
    eta_estdiff = polarization_ang_full_nan[:, 0, 0, None] - polarization_ang_full_nan[:, 0, :]

    p_mse = np.nanmean((p_pdiff - p_estdiff)**2)
    eta_mse = np.nanmean((eta_pdiff - eta_estdiff)**2)
    return p_mse, eta_mse


def generate_observations(zodipol, parser, n_rotations=40):
    motion_blur = 360 / n_rotations / zodipol.imager.exposure_time.value * u.deg
    obs_rot, rotation_list = get_measurements(zodipol, parser, n_rotations=n_rotations)
    obs_rot = [o.add_radial_blur(motion_blur, parser["resolution"]) for o in obs_rot]
    obs_rot = [o.add_direction_uncertainty(parser["fov"], parser["resolution"], parser["direction_uncertainty"]) for o in obs_rot]

    # create satellite polarizance and angle of polarization variables
    polarization_angle = parser["polarization_angle"]
    _, polarization_angle_spatial_diff = np.meshgrid(np.arange(parser["resolution"][0]),
                                                     np.deg2rad(np.linspace(-10, 10, parser["resolution"][1])),
                                                     indexing='ij')

    angular_amount = np.random.choice(np.linspace(-1, 1, n_rotations), size=(n_rotations), replace=False)
    pa_ts_diff = np.deg2rad(3) + polarization_angle_spatial_diff.flatten()[:, None, None][..., None] * angular_amount
    polarization_angle_real = polarization_angle[None, None, :, None] + pa_ts_diff

    polarizance, _ = np.meshgrid(np.linspace(-1, 0, parser["resolution"][0]), np.arange(parser["resolution"][1]),
                                 indexing='ij')
    polarizance_real = polarizance.reshape((len(obs_rot[0]), 1, 1))
    polariz_amount = np.random.choice(np.linspace(0, 0.4, n_rotations), size=(n_rotations), replace=False)
    polarizance_real = 0.9 + polarizance_real[..., None] * polariz_amount

    # create observations images
    obs_orig = [zodipol.make_camera_images(obs_rot[ii], polarizance_real[..., ii], polarization_angle_real[..., ii],
                                                 n_realizations=parser["n_realizations"], add_noise=True) for ii in range(n_rotations)]
    images_orig = np.stack(obs_orig, axis=-1)
    images_res = images_orig.reshape((parser["resolution"] + list(images_orig.shape[1:])))
    return images_res, rotation_list, polarizance_real, polarization_angle_real


def perform_estimation(zodipol, parser, rotation_list, images_res_flat, polarizance_real, polarization_angle_real,
                       disable=False, est_itr=6):
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
    for ii in tqdm(range(est_itr), disable=disable):
        # estimate observations
        obs_est_rot = estimate_observations(zodipol, parser, rotation_list, interp_images_res,
                                            polarizance_est_align, polarization_ang_rot, theta0, phi0)

        # estimate polarizance and angle of polarization
        est_pol = estimate_polarizance(obs_est_rot, images_res_flat.value, polarization_ang_full,
                                       zodipol, parser, disable=True)
        # est_pol = np.concatenate((init_polarizance[..., None], est_pol), axis=-1)
        polarizance_est_reshape = np.broadcast_to(est_pol.reshape((n_pixels, 1, n_rotations)),
                                                  polarization_ang_rot.shape)
        polarizance_est_align = align_images(zodipol, parser, polarizance_est_reshape, rotation_list)

        est_pa = estimate_polarization_angle(obs_est_rot, images_res_flat.value, polarizance_est,
                                             zodipol, parser, disable=True)
        # est_pa = np.concatenate((init_polarization_ang[..., None], est_pa), axis=-1)
        polarization_ang_est = est_pa.reshape((n_pixels, n_rotations))
        polarization_ang_full = polarization_ang_est[..., None, :] + polarization_angle[:, None]
        polarization_ang_align = align_images(zodipol, parser, polarization_ang_full, rotation_list)
        polarization_ang_rot = polarization_ang_align + np.deg2rad(rotation_list[None, None, :])

        est_images = get_images(obs_est_rot, polarization_ang_full, polarizance_est_reshape, zodipol, parser,
                                disable=True)
        cost = np.sqrt(np.nanmean(1e23 * (images_res_flat.value - est_images) ** 2))
        polarizance_cost = np.sqrt(np.sum((polarizance_real.squeeze() - polarizance_est_reshape[:, 0, :]) ** 2))
        polarization_ang_cost = np.sqrt(np.sum((polarization_angle_real.squeeze() - polarization_ang_full) ** 2))
        cost_itr['total'].append(cost)
        cost_itr['polarizance'].append(polarizance_cost)
        cost_itr['polarization_ang'].append(polarization_ang_cost)
    polarizance_est_reshape = np.where(nan_ind, np.nan, polarizance_est_reshape)
    polarization_ang_full = np.where(nan_ind, np.nan, polarization_ang_full)
    return cost_itr, polarizance_est_reshape, polarization_ang_full


def plot_estimation_comp(parser, polarizance_real, polarizance_est_reshape, polarization_angle_real, polarization_ang_full, saveto=None, set_colors=False, ii=-1):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    p1, p2 = polarizance_real.squeeze()[..., ii], polarizance_est_reshape[:, 0, ii]
    p3, p4 = polarization_angle_real.squeeze()[..., 0, ii], polarization_ang_full[:, 0, ii]
    if set_colors:
        v1, v2 = np.min(p1), np.max(p1)
        v3, v4 = np.min(p3), np.max(p3)
    else:
        v1, v2, v3, v4 = None, None, None, None
    c1 = ax1.imshow(p1.reshape(parser["resolution"]), vmin=v1, vmax=v2)
    c2 = ax2.imshow(p2.reshape(parser["resolution"]), vmin=v1, vmax=v2)
    c3 = ax3.imshow(p3.reshape(parser["resolution"]), vmin=v3, vmax=v4)
    c4 = ax4.imshow(p4.reshape(parser["resolution"]), vmin=v3, vmax=v4)
    cbar1 = fig.colorbar(c1, ax=ax1); cbar1.ax.tick_params(labelsize=14)
    cbar2 = fig.colorbar(c2, ax=ax2); cbar2.ax.tick_params(labelsize=14)
    cbar3 = fig.colorbar(c3, ax=ax3); cbar3.ax.tick_params(labelsize=14)
    cbar4 = fig.colorbar(c4, ax=ax4); cbar4.ax.tick_params(labelsize=14)
    ax1.set_title('Real Polarizance', fontsize=18); ax1.set_axis_off()
    ax2.set_title('Estimated Polarizance', fontsize=18); ax2.set_axis_off()
    ax3.set_title('Real Polarization Angle', fontsize=18); ax3.set_axis_off()
    ax4.set_title('Estimated Polarization Angle', fontsize=18); ax4.set_axis_off()
    fig.tight_layout()
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()


def plot_deviation_comp(parser, polarizance_real, polarizance_est_reshape, polarization_angle_real, polarization_ang_full, saveto=None, set_colors=False, ii=-1):
    pol_mean_deviation = polarizance_real.squeeze() - polarizance_real.mean(axis=-1).squeeze()[..., None]
    pol_est_mean_deviation = polarizance_est_reshape - polarizance_est_reshape.mean(axis=-1)[..., None]

    ang_mean_deviation = polarization_angle_real.squeeze() - polarization_angle_real.mean(axis=-1).squeeze()[..., None]
    ang_est_mean_deviation = polarization_ang_full - polarization_ang_full.mean(axis=-1)[..., None]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    p1, p2 = pol_mean_deviation.squeeze()[..., ii], pol_est_mean_deviation[:, 0, ii]
    p3, p4 = ang_mean_deviation.squeeze()[..., 0, ii], ang_est_mean_deviation[:, 0, ii]
    if set_colors:
        v1, v2 = np.min(p1), np.max(p1)
        v3, v4 = np.min(p3), np.max(p3)
    else:
        v1, v2, v3, v4 = None, None, None, None
    c1 = ax1.imshow(p1.reshape(parser["resolution"]), vmin=v1, vmax=v2)
    c2 = ax2.imshow(p2.reshape(parser["resolution"]), vmin=v1, vmax=v2)
    c3 = ax3.imshow(p3.reshape(parser["resolution"]), vmin=v3, vmax=v4)
    c4 = ax4.imshow(p4.reshape(parser["resolution"]), vmin=v3, vmax=v4)
    cbar1 = fig.colorbar(c1, ax=ax1); cbar1.ax.tick_params(labelsize=14)
    cbar2 = fig.colorbar(c2, ax=ax2); cbar2.ax.tick_params(labelsize=14)
    cbar3 = fig.colorbar(c3, ax=ax3); cbar3.ax.tick_params(labelsize=14)
    cbar4 = fig.colorbar(c4, ax=ax4); cbar4.ax.tick_params(labelsize=14)
    ax1.set_title('$P^{true}-P^{true}_{mean}$', fontsize=18); ax1.set_axis_off()
    ax2.set_title('$\hat{P}-\hat{P}_{mean}$', fontsize=18); ax2.set_axis_off()
    ax3.set_title('$\eta^{true}-\eta^{true}_{mean}$', fontsize=18); ax3.set_axis_off()
    ax4.set_title('$\hat{\eta}-\hat{\eta}_{mean}$', fontsize=18); ax4.set_axis_off()
    fig.tight_layout()
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()

def plot_cost(n_rotation_list, polariz_bias_l, pol_ang_bias_l, saveto=None, cost_type='MSE', xlabel='Number of observations'):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(n_rotation_list, polariz_bias_l)
    plt.grid()
    ax2 = ax.twinx()
    ax2.scatter(n_rotation_list, pol_ang_bias_l, color='r')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(f'Polarizance {cost_type}', color='C0', fontsize=16)
    ax2.set_ylabel(f'Polarization Angle {cost_type}', color='red', fontsize=16)
    ax.tick_params(axis='y', colors='C0', labelsize=14)
    ax2.tick_params(axis='y', colors='red', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    plt.title(f'{cost_type} of Polarizance and\nPolarization Angle Deviations Estimation', fontsize=18)
    fig.tight_layout()
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()


def main():
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    # generate observations
    polariz_bias_l, pol_ang_bias_l  = [], []
    rotation_cost_itr = []
    n_rotation_list = np.linspace(5, 21, 10).astype(int)
    # n_rotation_list = [21]
    for n_rotations in n_rotation_list:
        images_res, rotation_list, polarizance_real, polarization_angle_real = generate_observations(zodipol, parser, n_rotations=n_rotations)
        images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
        cost_itr, polarizance_est_reshape, polarization_ang_full = perform_estimation(zodipol, parser, rotation_list, images_res_flat, polarizance_real, polarization_angle_real)
        polarizance_cost, polarization_ang_cost = get_iteration_cost(polarizance_real, polarizance_est_reshape,
                                                                     polarization_angle_real, polarization_ang_full)
        polariz_bias_l.append(polarizance_cost)
        pol_ang_bias_l.append(polarization_ang_cost)
        rotation_cost_itr.append(cost_itr)

    xx = polarizance_real.squeeze().min(axis=0)
    xx = (max(xx) - xx) / (max(xx) - min(xx))
    ii = np.argmax(xx)

    # plot results
    plot_estimation_comp(parser, polarizance_real, polarizance_est_reshape, polarization_angle_real,
                         polarization_ang_full, saveto=None, ii=ii)
    plot_cost(n_rotation_list, polariz_bias_l, pol_ang_bias_l, saveto='outputs/self_calib_cost_n_obs.pdf', cost_type='Bias')

    plot_deviation_comp(parser, polarizance_real, polarizance_est_reshape, polarization_angle_real,
                        polarization_ang_full, saveto='outputs/self_estimation_comp.pdf', set_colors=True, ii=ii)
    pol_mean_deviation = polarizance_real.squeeze() - polarizance_real.mean(axis=-1).squeeze()[..., None]
    pol_est_mean_deviation = polarizance_est_reshape - polarizance_est_reshape.mean(axis=-1)[..., None]
    pol_dev_err = pol_mean_deviation - pol_est_mean_deviation[..., 0, :]

    ang_mean_deviation = polarization_angle_real.squeeze() - polarization_angle_real.mean(axis=-1).squeeze()[..., None]
    ang_est_mean_deviation = polarization_ang_full - polarization_ang_full.mean(axis=-1)[..., None]
    ang_dev_err = (ang_mean_deviation - ang_est_mean_deviation)[..., 0, :]
    plot_cost(xx, np.nanmean(pol_dev_err ** 2, axis=0), np.nanmean(ang_dev_err ** 2, axis=0), cost_type='MSE', saveto='outputs/self_estimation_mse.pdf', xlabel='Deviation Amount')
    pass


if __name__ == '__main__':
    main()
