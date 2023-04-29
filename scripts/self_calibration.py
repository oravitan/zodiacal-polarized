"""
This script performs self-calibration on a Zodipol object.
"""
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
from zodipol.zodipol import Zodipol, get_observations, get_initial_parameters


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
    obs_rot, rotation_list = get_observations(n_rotations, zodipol, parser)
    obs_rot = [o.add_radial_blur(motion_blur, list(parser["resolution"])) for o in obs_rot]
    obs_rot = [o.add_direction_uncertainty(parser["fov"], parser["resolution"], parser["direction_uncertainty"]) for o in obs_rot]

    # add birefringence
    delta_val, phi_val = np.pi / 4, np.pi / 6
    delta = zodipol.imager.get_birefringence_mat(delta_val, 'center', flat=True, inv=True)
    alpha = zodipol.imager.get_birefringence_mat(phi_val, 'linear', flat=True, angle=-np.pi / 4, min=0)
    mueller_truth = zodipol.imager.get_birefringence_mueller_matrix(delta, alpha)
    obs_biref = [zodipol.imager.apply_birefringence(o, mueller_truth) for o in obs_rot]

    # create satellite polarizance and angle of polarization variables
    polarization_angle = parser["polarization_angle"]
    _, polarization_angle_spatial_diff = np.meshgrid(np.arange(parser["resolution"][0]),
                                                     np.deg2rad(np.linspace(-10, 10, parser["resolution"][1])),
                                                     indexing='ij')

    pa_ts_diff = np.zeros_like(polarization_angle_spatial_diff.flatten()[..., None])
    polarization_angle_real = polarization_angle[None, :] + pa_ts_diff

    polarizance, _ = np.meshgrid(np.linspace(-1, 0, parser["resolution"][0]), np.arange(parser["resolution"][1]),
                                 indexing='ij')
    polarizance_real = polarizance.reshape((len(obs_biref[0]), 1))
    polarizance_real = 0.95 + 0.35 * polarizance_real

    # create observations images
    obs_orig = [zodipol.make_camera_images(obs_biref[ii], polarizance_real[..., None, :], polarization_angle_real[..., None, :],
                                                 n_realizations=parser["n_realizations"], add_noise=True) for ii in range(n_rotations)]  #
    images_orig = np.stack(obs_orig, axis=-1)
    images_res = images_orig.reshape((parser["resolution"] + list(images_orig.shape[1:])))
    return obs_rot, images_res, rotation_list, polarizance_real.squeeze(), pa_ts_diff.squeeze(), delta, alpha


def perform_estimation(zodipol, parser, rotation_list, images_res_flat, polarizance_real, polarization_angle_real, mueller_truth, n_itr=10):
    theta0, phi0 = zodipol.create_sky_coords(theta=parser["direction"][0], phi=parser["direction"][1], roll=0 * u.deg, resolution=parser["resolution"])
    callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real, mueller=mueller_truth)
    self_calib = SelfCalibration(images_res_flat, rotation_list, zodipol, parser, theta=theta0, phi=phi0)
    init_dict = {'eta': polarization_angle_real}
    cost_itr, clbk_itr = self_calib.calibrate(images_res_flat, n_itr=n_itr, callback=callback_partial, init=init_dict)
    p, eta, biref = self_calib.get_properties()
    return cost_itr, p, eta, biref, clbk_itr


def cost_callback(calib: SelfCalibration, p, eta, mueller):
    cp, _, biref = calib.get_properties()
    p_cost = np.nanmean((p - np.mean(p) - (cp - np.nanmean(cp))) ** 2)
    mueller_cost = np.nanmean((mueller[..., 1:3, 1:3] - biref[..., 1:3, 1:3])**2)
    return p_cost, mueller_cost


def plot_deviation_comp(parser, polarizance_real, polarizance_est_reshape, saveto=None, set_colors=False):
    pol_mean_deviation = polarizance_real
    pol_est_mean_deviation = polarizance_est_reshape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    p1, p2 = pol_mean_deviation.squeeze(), pol_est_mean_deviation

    c1 = ax1.imshow(p1.reshape(parser["resolution"]))
    c2 = ax2.imshow(p2.reshape(parser["resolution"]))
    cbar1 = fig.colorbar(c1, ax=ax1); cbar1.ax.tick_params(labelsize=14)
    cbar2 = fig.colorbar(c2, ax=ax2); cbar2.ax.tick_params(labelsize=14)
    ax1.set_title('True $P$', fontsize=18); ax1.set_axis_off()
    ax2.set_title('$\hat{P}$', fontsize=18); ax2.set_axis_off()
    fig.tight_layout()
    if saveto is not None:
        plt.savefig(saveto, format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
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
        plt.savefig(saveto, format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()

def plot_mueller(mueller, parser, cbar=False, saveto=None, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanmin(mueller)
    if vmax is None:
        vmax = np.nanmax(mueller)
    mueller = mueller[..., :3, :3]
    fig, ax = plt.subplots(3,3, figsize=(6,6), sharex='col', sharey='row', subplot_kw={'xticks': [], 'yticks': []})
    for i in range(3):
        for j in range(3):
            c = ax[i,j].imshow(mueller[..., i, j].reshape(parser["resolution"]), vmin=vmin, vmax=vmax)
            # ax[i,j].get_xaxis().set_visible(False)
            # ax[i,j].get_yaxis().set_visible(False)
    ax[0,0].set_ylabel(0, fontsize=16)
    ax[1,0].set_ylabel(1, fontsize=16)
    ax[2,0].set_ylabel(2, fontsize=16)
    ax[2,0].set_xlabel(0, fontsize=16)
    ax[2,1].set_xlabel(1, fontsize=16)
    ax[2,2].set_xlabel(2, fontsize=16)
    # fig.colorbar(c, ax=ax.ravel().tolist())
    if cbar:
        cb = fig.colorbar(c, ax=ax.ravel().tolist())
        cb.ax.tick_params(labelsize=14)
    else:
        plt.tight_layout(w_pad=-15.0, h_pad=1.0)
    if saveto is not None:
        plt.savefig(saveto, format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()


def plot_cost_itr(cost_itr, p_cost, mueller_cost, saveto=None):
    fig, ax = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
    ax[0].plot(cost_itr, lw=3)
    ax[0].set_ylabel('Intensity MSE\n($electrons^2$)', fontsize=16)
    ax[0].tick_params(labelsize=16)
    ax[0].grid()

    ax[1].plot(p_cost, lw=3)
    ax[1].set_ylabel('$\hat{P}$ MSE', fontsize=16)
    ax[1].tick_params(labelsize=16)
    ax[1].grid()

    ax[2].plot(mueller_cost, lw=3)
    ax[2].set_ylabel('$\hat{B}$ MSE', fontsize=16)
    ax[2].tick_params(labelsize=16)
    ax[2].grid()
    ax[2].set_xlabel('Iteration number', fontsize=16)
    fig.tight_layout()
    if saveto is not None:
        plt.savefig(saveto, format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()


def main():
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])
    n_itr = 20

    # generate observations
    n_rotations = 20
    obs_truth, rotation_list = get_observations(n_rotations, zodipol, parser)
    obs_truth, images_res, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(obs_truth, parser, zodipol)
    images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
    images_res_flat = zodipol.post_process_images(images_res_flat)
    cost_itr, p_hat, eta_hat, biref, clbk_itr = perform_estimation(zodipol, parser, rotation_list, images_res_flat,
                                                  polarizance_real, polarization_angle_real, mueller_truth, n_itr=n_itr)
    p_cost, mueller_cost = list(zip(*clbk_itr))
    plot_cost_itr(cost_itr, p_cost, mueller_cost, saveto='outputs/self_calibration_cost_itr.pdf')
    plot_deviation_comp(parser, polarizance_real, p_hat, set_colors=True, saveto='outputs/self_calibration_polarizance_est.pdf')
    plot_mueller(biref, parser, cbar=True, vmin=-0.05, vmax=1, saveto='outputs/self_calib_birefringence_est.pdf')
    pass


if __name__ == '__main__':
    main()
