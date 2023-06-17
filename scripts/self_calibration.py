"""
This script performs self-calibration on a Zodipol object.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from tqdm import tqdm
from functools import partial
from typing import List
from datetime import datetime

# import the necessary modules
from zodipol.estimation.self_calibration import SelfCalibration
from zodipol.estimation.calibration import Calibration
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol.zodipol import Zodipol
from zodipol.zodipol.generate import get_observations, get_initial_parameters, get_initialization
from zodipol.visualization.calibration_plots import plot_deviation_comp, plot_mueller, plot_cost_itr, \
    plot_res_comp_plot, plot_all_calibration_props, compare_self_and_calib


run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
outputs_dir = f'outputs/{run_time}'
os.mkdir(outputs_dir)


def calibrate(obs_truth, zodipol: Zodipol, parser: ArgParser, images_res_flat: np.ndarray,
                   polarizance_real: np.ndarray, polarization_angle_real: np.ndarray, mueller_truth: np.ndarray,
                   n_itr: int = 10, disable=False, initialization=None, **kwargs):
    """
    Perform self-calibration on a Zodipol object.
    """
    obs_comb = zodipol.combine_observations(obs_truth, polarizance=polarizance_real.squeeze(),
                                            polarization_angle=polarization_angle_real)
    true_values = {'images': images_res_flat, 'p': polarizance_real, 'eta': polarization_angle_real, 'biref': mueller_truth}
    calib = Calibration(obs_comb, zodipol, parser)
    init_dict = {'eta': polarization_angle_real, **initialization}
    callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real, mueller=mueller_truth)
    p, eta, biref, cost, itr_cost = calib.calibrate(images_res_flat, n_itr=n_itr, callback=callback_partial, init=init_dict,
                                                    disable=disable, **kwargs)
    est_images = np.stack([calib.forward_model(o) for o in obs_comb], axis=-1)
    est_values = {'images': est_images, 'p': p, 'eta': eta, 'biref': biref}
    return cost, est_values, itr_cost


def self_calibrate(zodipol: Zodipol, parser: ArgParser, rotation_list: List[int], images_res_flat: np.ndarray,
                   polarizance_real: np.ndarray, polarization_angle_real: np.ndarray, mueller_truth: np.ndarray,
                   n_itr: int = 10, disable=False, initialization=None, **kwargs):
    """
    Perform self-calibration on a Zodipol object.
    """
    theta0, phi0 = zodipol.create_sky_coords(theta=parser["direction"][0], phi=parser["direction"][1], roll=0 * u.deg, resolution=parser["resolution"])
    callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real, mueller=mueller_truth)
    self_calib = SelfCalibration(images_res_flat, rotation_list, zodipol, parser, theta=theta0, phi=phi0)
    init_dict = {'eta': polarization_angle_real, **initialization}
    _, _, _, cost_itr, clbk_itr = self_calib.calibrate(images_res_flat, n_itr=n_itr, callback=callback_partial,
                                                       init=init_dict, disable=disable, **kwargs)
    images, p, eta, biref = self_calib.get_properties()
    est_values = {'images': images, 'p': p, 'eta': eta, 'biref': biref}
    return cost_itr, est_values, clbk_itr


def run_self_calibration(n_rotations, n_itr, zodipol, parser, disable=False, direction_uncertainty=None,
                         self_calibration_flag=True, **kwargs):
    obs_truth, rotation_list = get_observations(n_rotations, zodipol, parser)
    obs_truth, images_res, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(
        obs_truth, parser, zodipol, mode='sine', direction_uncertainty=direction_uncertainty)
    images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
    images_res_flat = zodipol.post_process_images(images_res_flat)
    initialization = get_initialization(polarizance_real, mueller_truth)
    star_pixels = np.stack([o.star_pixels for o in obs_truth], axis=-1)
    if self_calibration_flag:
        cost_itr, est_values, clbk_itr = self_calibrate(zodipol, parser, rotation_list, images_res_flat,
                                                        polarizance_real, polarization_angle_real,
                                                        mueller_truth, n_itr=n_itr, disable=disable, max_p=np.max(polarizance_real),
                                                        initialization=initialization, star_pixels=star_pixels, **kwargs)
    else:
        cost_itr, est_values, clbk_itr = calibrate(obs_truth, zodipol, parser, images_res_flat, polarizance_real,
                                                   polarization_angle_real, mueller_truth, n_itr=n_itr, disable=disable,
                                                   initialization=initialization, star_pixels=star_pixels, **kwargs)
    true_values = {'images': images_res_flat, 'p': polarizance_real, 'eta': polarization_angle_real, 'biref': mueller_truth}
    return cost_itr, est_values, true_values, clbk_itr


def cost_callback(calib: SelfCalibration, p: np.ndarray, eta: np.ndarray, mueller: np.ndarray):
    """
    Callback function to calculate the cost function.
    """
    _, cp, _, biref = calib.get_properties()
    p_cost = np.nanmean((p - cp) ** 2) ** 0.5
    mueller_cost = np.nanmean((mueller[..., 1:3, 1:3] - biref[..., 1:3, 1:3])**2) ** 0.5
    p_std = np.nanstd(p - cp)
    mueller_std = np.nanstd(mueller[..., 1:3, 1:3] - biref[..., 1:3, 1:3])
    p_mad = np.nanmedian(np.abs(p - cp))
    mueller_mad = np.nanmedian(np.abs(mueller[..., 1:3, 1:3] - biref[..., 1:3, 1:3]))
    return p_cost, mueller_cost, p_std, mueller_std, p_mad, mueller_mad


def main_show_cost(n_rotations=10, n_itr=10, name='self_calib', **kwargs):
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    # generate observations
    cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser, normalize_eigs=True, kernel_size=5, **kwargs)
    p_cost, mueller_cost, p_std, mueller_std, p_mad, mueller_mad = list(zip(*clbk_itr))
    plot_cost_itr(cost_itr, p_cost, mueller_cost, saveto=f'{outputs_dir}/self_calibration_cost_itr.pdf')
    plot_deviation_comp(parser, true_values["p"][..., 0], est_values['p'][..., 0],
                        saveto=f'{outputs_dir}/{name}_polarizance_est.pdf')
    plot_mueller(est_values['biref'] - np.eye(3)[None, ...], parser, cbar=True, saveto=f'{outputs_dir}/{name}_birefringence_est.pdf')
    plot_all_calibration_props(true_values["p"][..., 0], true_values["biref"], parser["resolution"], saveto=f'{outputs_dir}/{name}_true_values.pdf')
    plot_all_calibration_props(est_values["p"][..., 0], est_values["biref"], parser["resolution"],
                               saveto=f'{outputs_dir}/{name}_est_values.pdf')


def compare_calib_self_calib(n_rotations=10, n_itr=10, **kwargs):
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    # generate observations
    self_cost_itr, self_est_values, self_true_values, self_clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser, self_calibration_flag=True,
                                                                       normalize_eigs=True, kernel_size=5, **kwargs)
    cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser,
                                                                       self_calibration_flag=False, normalize_eigs=True, kernel_size=5, **kwargs)
    compare_self_and_calib(true_values['p'][:, 0], self_est_values['p'][:, 0], est_values['p'][:, 0],
                           xlabel='true P', ylabel='$\hat{P}$', saveto=f'{outputs_dir}/compare_calib_self_calib.pdf')

def main_plot_n_obs(n_itr=10, n_rotations_list=None, **kwargs):
    if n_rotations_list is None:
        n_rotations_list = np.linspace(10, 30, 10, endpoint=True, dtype=int)
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    # generate observations
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())
    res_cost = []
    for n_rotations in tqdm(n_rotations_list):
        n_rot_res = []
        for ii in range(3):
            cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser,
                                                                               disable=True, normalize_eigs=True, kernel_size=5, **kwargs)
            mean_num_electrons = np.mean((true_values["images"] / A_gamma).to('').value)
            n_rot_res.append((cost_itr[-1] / mean_num_electrons,) + clbk_itr[-1])
        res_cost.append(n_rot_res)
    res_cost_arr = np.array(res_cost)
    rot_p_mse, p_err = np.mean(res_cost_arr[..., 1], axis=1), np.std(res_cost_arr[..., 1], axis=1)
    rot_biref_mse, biref_err = np.mean(res_cost_arr[..., 2], axis=1), np.std(res_cost_arr[..., 2], axis=1)
    plot_res_comp_plot(n_rotations_list, rot_p_mse, rot_biref_mse, saveto=f"{outputs_dir}/calib_mse_n_rotations.pdf",
                       xlabel="K", ylim1=(0, None), ylim2=(0, None), p_mse_err=p_err, biref_mse_err=biref_err)
    return n_rotations_list, res_cost


def main_plot_exp_time(n_rotations=30, n_itr=10, exposure_time_list=None, **kwargs):
    if exposure_time_list is None:
        exposure_time_list = np.logspace(np.log10(10), np.log10(60), 10)
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())

    # now estimate how well we calibration based on exposure time
    res_cost = []
    for exposure_time in tqdm(exposure_time_list):
        n_ex_res = []
        for ii in range(3):
            zodipol.imager.exposure_time = exposure_time * u.s
            cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser,
                                                                               disable=True, normalize_eigs=True, kernel_size=5, **kwargs)
            mean_num_electrons = np.mean((true_values["images"] / A_gamma).to('').value)
            n_ex_res.append((cost_itr[-1] / mean_num_electrons,) + clbk_itr[-1])
        res_cost.append(n_ex_res)
    res_cost_arr = np.array(res_cost)
    ex_p_mse, p_err = np.mean(res_cost_arr[..., 1], axis=1), np.std(res_cost_arr[..., 1], axis=1)
    ex_biref_mse, biref_err = np.mean(res_cost_arr[..., 2], axis=1), np.std(res_cost_arr[..., 2], axis=1)

    plot_res_comp_plot(exposure_time_list, ex_p_mse, ex_biref_mse, saveto=f"{outputs_dir}/calib_mse_exposure_time.pdf",
                       xlabel="$\Delta t \;(s)$", ylim1=(0, None), ylim2=(0, None), p_mse_err=p_err, biref_mse_err=biref_err)
    return exposure_time_list, res_cost


def main_plot_uncertainty(n_rotations=10, n_itr=10, direction_error_list=None, **kwargs):
    if direction_error_list is None:
        direction_error_list = np.logspace(np.log10(0.005), np.log10(2.5), 10)
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    # now estimate how well we calibration based on direction uncertainty
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())
    res_cost = []
    for direction_error in tqdm(direction_error_list):
        dir_err = []
        for ii in range(3):
            cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser,
                                                                               disable=True, normalize_eigs=True,
                                                                               kernel_size=5,
                                                                               direction_uncertainty=direction_error * u.deg, **kwargs)
            mean_num_electrons = np.mean((true_values["images"] / A_gamma).to('').value)
            dir_err.append((cost_itr[-1] / mean_num_electrons,) + clbk_itr[-1])
        res_cost.append(dir_err)
    res_cost_arr = np.array(res_cost)
    dir_p_mse, p_err = np.mean(res_cost_arr[..., 1], axis=1), np.std(res_cost_arr[..., 1], axis=1)
    dir_biref_mse, biref_err = np.mean(res_cost_arr[..., 2], axis=1), np.std(res_cost_arr[..., 2], axis=1)

    plot_res_comp_plot(direction_error_list, dir_p_mse, dir_biref_mse, saveto=f"{outputs_dir}/calib_mse_direction_error.pdf",
                       xlabel="Direction Error (deg)", ylim1=(0, None), ylim2=(0, None), p_mse_err=p_err, biref_mse_err=biref_err)
    return direction_error_list, res_cost


def main():
    main_show_cost(n_rotations=30, n_itr=5, self_calibration_flag=True)
    compare_calib_self_calib(n_rotations=30, n_itr=5)
    n_obs_list, cost_n_obs = main_plot_n_obs(n_itr=5)
    exp_t_list, cost_expo = main_plot_exp_time(n_rotations=30, n_itr=5)
    dir_unc_list, cost_dir_unc = main_plot_uncertainty(n_rotations=30, n_itr=5)
    pass


if __name__ == '__main__':
    main()
