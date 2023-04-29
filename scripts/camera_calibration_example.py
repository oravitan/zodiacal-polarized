import os
import pickle as pkl
import logging
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from functools import partial

# import the necessary modules
from zodipol.estimation.calibration import Calibration
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol, get_observations, get_initial_parameters

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def cost_callback(calib: Calibration, p, eta, mueller):
    mueller_est = calib.biref
    p_cost = np.nanmean((p.squeeze() - calib.p)**2)
    mueller_cost = np.nanmean((mueller[..., 1:3, 1:3] - mueller_est[..., 1:3, 1:3])**2)
    return p_cost, mueller_cost


def plot_mueller(mueller, cbar=False, saveto=None):
    mueller = mueller[..., :3, :3]
    fig, ax = plt.subplots(3,3, figsize=(6,6), sharex='col', sharey='row', subplot_kw={'xticks': [], 'yticks': []})
    for i in range(3):
        for j in range(3):
            c = ax[i,j].imshow(mueller[..., i, j].reshape(parser["resolution"]), vmin=mueller.min(), vmax=mueller.max())
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


if __name__ == '__main__':
    # set params
    logging.info(f'Started run.')
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    n_rotations = 20
    n_itr = 10
    obs, rotation_list = get_observations(n_rotations, zodipol, parser)

    obs, images_orig, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(obs, parser, zodipol)
    obs_comb = zodipol.combine_observations(obs, polarizance=polarizance_real.squeeze(), polarization_angle=polarization_angle_real)
    callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real, mueller=mueller_truth)

    calib = Calibration(obs_comb, zodipol, parser)
    init = {'eta': polarization_angle_real}
    p, eta, biref, cost, itr_cost = calib.calibrate(images_orig, n_itr=n_itr, callback=callback_partial, init=init, normalize_eigs=True)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 5))
    for ax_i, c in zip(ax, [cost] + list(zip(*itr_cost))):
        ax_i.semilogy(c, lw=3)
        ax_i.grid()
        ax_i.tick_params(labelsize=14)
    ax[-1].set_xlabel("Iteration", fontsize=16)
    ax[0].set_ylabel("Intensity MSE\n($electrons^2$)", fontsize=16)
    ax[1].set_ylabel("$\hat{P}$ MSE", fontsize=16)
    ax[2].set_ylabel("$\hat{B}$ MSE", fontsize=16)
    # plt.suptitle("Cost vs. Iteration", fontsize=18)
    plt.tight_layout()
    plt.savefig("outputs/calib_cost_vs_iteration.pdf", format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()

    # plot comparison of P,eta estimated vs true
    fig, ax = plt.subplots(1, 2, figsize=(5, 3), subplot_kw={'xticks': [], 'yticks': []})
    c1 = ax[0].imshow(polarizance_real[:, 0].reshape((parser["resolution"][0], parser["resolution"][1])), vmin=0.7, vmax=1)
    ax[0].set_title("$P^{true}$")
    plt.colorbar(c1, ax=ax[0])
    c2 = ax[1].imshow(p[:, 0].reshape((parser["resolution"][0], parser["resolution"][1])), vmin=0.7, vmax=1)
    ax[1].set_title("$\hat{P}$")
    plt.colorbar(c2, ax=ax[1])
    plt.savefig("outputs/calib_p_eta.pdf", format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()

    plot_mueller(biref, cbar=True, saveto="outputs/calib_mueller_estimation.pdf")

    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())

    # calibration vs. number of observations
    n_rotations_list = [4, 6, 10, 14, 18, 22, 26, 30]
    res_cost = []
    for n_rotations in n_rotations_list:
        obs, rotation_list = get_observations(n_rotations)
        obs, images_orig, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(obs, parser,
                                                                                                            zodipol)
        obs_comb = zodipol.combine_observations(obs, polarizance=polarizance_real.squeeze(),
                                                polarization_angle=polarization_angle_real)
        callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real,
                                   mueller=mueller_truth)

        calib = Calibration(obs_comb, zodipol, parser)
        init = {'eta': polarization_angle_real}
        p, eta, biref, clbk_itr, itr_cost = calib.calibrate(images_orig, n_itr=n_itr,
                                                            callback=callback_partial, init=init)
        p_cost, mueller_cost = list(zip(*itr_cost))
        mean_num_electrons = np.mean((images_orig / A_gamma).to('').value)
        res_cost.append((clbk_itr[-1] / mean_num_electrons, p_cost[-1], mueller_cost[-1]))
    rot_intensity_mse, rot_p_mse, rot_biref_mse = list(zip(*res_cost))

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.semilogy(n_rotations_list, rot_p_mse, lw=2, c='b')
    ax2 = ax.twinx()
    ax2.semilogy(n_rotations_list, rot_biref_mse, lw=2, c='r')
    ax.grid()
    ax.set_xlabel("Number of observations", fontsize=16)
    ax.set_ylabel("$P$ MSE", fontsize=16, c='b')
    ax2.set_ylabel("${\\bf B}$ MSE", fontsize=16, c='r')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', colors='b', labelsize=16)
    ax2.tick_params(axis='y', colors='r', labelsize=16)
    fig.tight_layout()
    plt.savefig("outputs/calib_mse_n_rotations.pdf", format='pdf', bbox_inches='tight', transparent="True",
                pad_inches=0)
    plt.show()


    # now estimate how well we calibration based on exposure time
    n_rotations = 20
    n_itr = 10
    exposure_time_list = np.logspace(np.log10(0.5), np.log10(100), 20)
    res_cost = []
    for exposure_time in exposure_time_list:
        zodipol.imager.exposure_time = exposure_time * u.s
        obs, rotation_list = get_observations(n_rotations)
        obs, images_orig, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(obs, parser, zodipol)
        obs_comb = zodipol.combine_observations(obs, polarizance=polarizance_real.squeeze(), polarization_angle=polarization_angle_real)
        callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real, mueller=mueller_truth)

        calib = Calibration(obs_comb, zodipol, parser)
        init = {'eta': polarization_angle_real}
        p, eta, biref, clbk_itr, itr_cost = calib.calibrate(images_orig, n_itr=n_itr, callback=callback_partial,init=init)
        p_cost, mueller_cost = list(zip(*itr_cost))
        mean_num_electrons = np.mean((images_orig / A_gamma).to('').value)
        res_cost.append((clbk_itr[-1]/mean_num_electrons, p_cost[-1], mueller_cost[-1]))
    ex_intensity_mse, ex_p_mse, ex_biref_mse = list(zip(*res_cost))

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.loglog(exposure_time_list, ex_p_mse, lw=2, c='b')
    ax2 = ax.twinx()
    ax2.loglog(exposure_time_list, ex_biref_mse, lw=2, c='r')
    ax.grid()
    ax.set_xlabel("$\Delta t \;(s)$", fontsize=16)
    ax.set_ylabel("$P$ MSE", fontsize=16, c='b')
    ax2.set_ylabel("${\\bf B}$ MSE", fontsize=16, c='r')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', colors='b', labelsize=16)
    ax2.tick_params(axis='y', colors='r', labelsize=16)
    fig.tight_layout()
    plt.savefig("outputs/calib_mse_exposure_time.pdf", format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()

    fig, (ax, ax2) = plt.subplots(2,1, figsize=(6, 6))
    ax.semilogy(n_rotations_list, rot_p_mse, lw=2, c='b')
    ax_d = ax.twinx()
    ax_d.semilogy(n_rotations_list, rot_biref_mse, lw=2, c='r')
    ax.grid()
    ax.set_xlabel("Number of observations", fontsize=16)
    ax.set_ylabel("$P$ MSE", fontsize=16, c='b')
    ax_d.set_ylabel("${\\bf B}$ MSE", fontsize=16, c='r')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', colors='b', labelsize=16)
    ax_d.tick_params(axis='y', colors='r', labelsize=16)
    ax_d.yaxis.set_tick_params(which='major', right=False, left=False)
    plt.setp(ax_d.get_yminorticklabels(), visible=False)

    ax2.semilogy(exposure_time_list, ex_p_mse, lw=2, c='b')
    ax_d2 = ax2.twinx()
    ax_d2.semilogy(exposure_time_list, ex_biref_mse, lw=2, c='r')
    ax2.grid()
    ax2.set_xlabel("$\Delta t \;(s)$", fontsize=16)
    ax2.set_ylabel("$P$ MSE", fontsize=16, c='b')
    ax_d2.set_ylabel("${\\bf B}$ MSE", fontsize=16, c='r')
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', colors='b', labelsize=16)
    ax_d2.tick_params(axis='y', colors='r', labelsize=16)
    fig.tight_layout()
    plt.savefig("outputs/calib_mse_exposure_nobs.pdf", format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()
