import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from tqdm import tqdm
from functools import partial

# import the necessary modules
from zodipol.estimation.self_calibration import SelfCalibration
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol


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
    # obs_rot = [o.add_radial_blur(motion_blur, list(parser["resolution"])) for o in obs_rot]
    # obs_rot = [o.add_direction_uncertainty(parser["fov"], parser["resolution"], parser["direction_uncertainty"]) for o in obs_rot]

    # create satellite polarizance and angle of polarization variables
    polarization_angle = parser["polarization_angle"]
    _, polarization_angle_spatial_diff = np.meshgrid(np.arange(parser["resolution"][0]),
                                                     np.deg2rad(np.linspace(-10, 10, parser["resolution"][1])),
                                                     indexing='ij')

    # angular_amount = np.random.choice(np.linspace(-1, 1, n_rotations), size=(n_rotations), replace=False)
    angular_amount = np.linspace(0, 0, n_rotations)
    # angular_amount = np.linspace(-1, 1, n_rotations)
    pa_ts_diff = polarization_angle_spatial_diff.flatten()[:, None, None][..., None] * angular_amount
    # pa_ts_diff = np.deg2rad(3) + polarization_angle_spatial_diff.flatten()[:, None, None][..., None] * angular_amount
    polarization_angle_real = polarization_angle[None, None, :, None] + pa_ts_diff

    polarizance, _ = np.meshgrid(np.linspace(-1, 0, parser["resolution"][0]), np.arange(parser["resolution"][1]),
                                 indexing='ij')
    polarizance_real = polarizance.reshape((len(obs_rot[0]), 1, 1))
    # polariz_amount = np.random.choice(np.linspace(0, 0.4, n_rotations), size=(n_rotations), replace=False)
    polariz_amount = np.linspace(0, 0.4, n_rotations)
    # polariz_amount = np.linspace(0, 0, n_rotations)  # TODO: remove this line
    polarizance_real = 0.9 + polarizance_real[..., None] * polariz_amount

    # create observations images
    obs_orig = [zodipol.make_camera_images(obs_rot[ii], polarizance_real[..., ii], polarization_angle_real[..., ii],
                                                 n_realizations=parser["n_realizations"], add_noise=True) for ii in range(n_rotations)]
    images_orig = np.stack(obs_orig, axis=-1)
    images_res = images_orig.reshape((parser["resolution"] + list(images_orig.shape[1:])))
    return images_res, rotation_list, polarizance_real.squeeze(), pa_ts_diff.squeeze()


def perform_estimation(zodipol, parser, rotation_list, images_res_flat, polarizance_real, polarization_angle_real, n_itr=10):
    theta0, phi0 = zodipol._create_sky_coords(theta=parser["direction"][0], phi=parser["direction"][1], roll=0 * u.deg, resolution=parser["resolution"])
    callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real, mueller=0)
    self_calib = SelfCalibration(images_res_flat, rotation_list, zodipol, parser, theta=theta0, phi=phi0)
    p, eta, _, _, cost_itr, clbk_itr = self_calib.calibrate(n_itr=n_itr, mode="P,eta", callback=callback_partial)

    interp_images_res = self_calib.align_images(images_res_flat.value, rotation_list)
    nan_ind = np.isnan(interp_images_res).any(axis=-2).any(axis=-1, keepdims=True)
    polarizance_est_reshape = np.where(nan_ind, np.nan, p)
    polarization_ang_full = np.where(nan_ind, np.nan, eta)
    return cost_itr, polarizance_est_reshape, polarization_ang_full


def cost_callback(calib: SelfCalibration, p, eta, mueller):
    mueller_est = calib.zodipol.imager.get_birefringence_mueller_matrix(calib.delta, calib.alpha)
    p_cost = np.mean((p - calib.p)**2)
    eta_cost = np.mean((eta - calib.eta)**2)
    mueller_cost = np.mean((mueller - mueller_est)**2)
    return p_cost, eta_cost, mueller_cost


def plot_deviation_comp(parser, polarizance_real, polarizance_est_reshape, polarization_angle_real, polarization_ang_full, saveto=None, set_colors=False, ii=-1):
    pol_mean_deviation = polarizance_real.squeeze() - polarizance_real.mean(axis=-1).squeeze()[..., None]
    pol_est_mean_deviation = polarizance_est_reshape - polarizance_est_reshape.mean(axis=-1)[..., None]

    ang_mean_deviation = polarization_angle_real.squeeze() - polarization_angle_real.mean(axis=-1).squeeze()[..., None]
    ang_est_mean_deviation = polarization_ang_full - polarization_ang_full.mean(axis=-1)[..., None]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    p1, p2 = pol_mean_deviation.squeeze()[..., ii], pol_est_mean_deviation[:,ii]
    p3, p4 = ang_mean_deviation.squeeze()[..., ii], ang_est_mean_deviation[:,ii]
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
    n_itr = 5

    # generate observations
    rotation_cost_itr = []
    # n_rotation_list = np.linspace(5, 21, 10).astype(int)
    n_rotation_list = [9]  # TODO: remove this
    for n_rotations in n_rotation_list:
        images_res, rotation_list, polarizance_real, polarization_angle_real = generate_observations(zodipol, parser, n_rotations=n_rotations)
        images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
        cost_itr, p_hat, eta_hat = perform_estimation(zodipol, parser, rotation_list, images_res_flat,
                                                      polarizance_real, polarization_angle_real, n_itr=n_itr)
        rotation_cost_itr.append(cost_itr)

        fig, ax = plt.subplots(1,1)
        plt.plot(cost_itr, lw=3)
        plt.xlabel('Iteration number', fontsize=16)
        plt.ylabel('Iteration MSE', fontsize=16)
        ax.tick_params(labelsize=16)
        plt.grid()
        fig.tight_layout()
        plt.show()

        for ii in range(n_rotations):
            plot_deviation_comp(parser, polarizance_real, p_hat, polarization_angle_real,
                                eta_hat, set_colors=True, ii=ii)
        pol_mean_deviation = polarizance_real.squeeze() - np.nanmean(polarizance_real, axis=-1, keepdims=True)
        pol_est_mean_deviation = p_hat - np.nanmean(p_hat, axis=-1, keepdims=True)
        pol_dev_err = pol_mean_deviation - pol_est_mean_deviation

        ang_mean_deviation = polarization_angle_real.squeeze() - np.nanmean(polarization_angle_real, axis=-1, keepdims=True)
        ang_est_mean_deviation = eta_hat - np.nanmean(eta_hat, axis=-1, keepdims=True)
        ang_dev_err = ang_mean_deviation - ang_est_mean_deviation
        plot_cost(rotation_list, np.nanmean(pol_dev_err ** 2, axis=0), np.nanmean(ang_dev_err ** 2, axis=0), cost_type='MSE', saveto='outputs/self_estimation_mse.pdf', xlabel='Deviation Amount')
        pass


if __name__ == '__main__':
    main()
