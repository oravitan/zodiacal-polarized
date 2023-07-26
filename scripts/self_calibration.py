"""
This script performs self-calibration on a Zodipol object.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import multiprocessing as mp

from tqdm import tqdm
from functools import partial
from typing import List
from datetime import datetime
from itertools import repeat

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
    self_calib = SelfCalibration(images_res_flat, rotation_list, zodipol, parser, theta=theta0, phi=phi0, min_num_samples=5)
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


def main_show_cost(parser, cost_itr, est_values, true_values, clbk_itr, name='self_calib', p_kwargs=None, a_kwargs=None, b_kwargs=None, c_kwargs=None):
    p_cost, mueller_cost, p_std, mueller_std, p_mad, mueller_mad = list(zip(*clbk_itr))
    plot_cost_itr(cost_itr, p_cost, mueller_cost, saveto=f'{outputs_dir}/self_calibration_cost_itr.pdf')
    # plot_deviation_comp(parser, true_values["p"][..., 0], est_values['p'][..., 0],
    #                     saveto=f'{outputs_dir}/{name}_polarizance_est.pdf')
    # plot_mueller(est_values['biref'] - np.eye(3)[None, ...], parser, cbar=True, saveto=f'{outputs_dir}/{name}_birefringence_est.pdf')

    plot_all_calibration_props(true_values["p"][..., 0], true_values["biref"], parser["resolution"], saveto=f'{outputs_dir}/{name}_true_values.pdf', p_kwargs=p_kwargs, a_kwargs=a_kwargs,
                               b_kwargs=b_kwargs, c_kwargs=c_kwargs)
    plot_all_calibration_props(est_values["p"][..., 0], est_values["biref"], parser["resolution"],
                               saveto=f'{outputs_dir}/{name}_est_values.pdf', p_kwargs=p_kwargs, a_kwargs=a_kwargs,
                               b_kwargs=b_kwargs, c_kwargs=c_kwargs)
    print(f'{name} results: polarizance cost: {p_cost[-1]:.3f}, mueller cost: {mueller_cost[-1]:.3f}')
    pass


def compare_calib_self_calib(n_rotations=10, n_itr=10, **kwargs):
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"], solar_cut=5 * u.deg)

    # generate observations
    self_cost_itr, self_est_values, self_true_values, self_clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser, self_calibration_flag=True,
                                                                       normalize_eigs=True, kernel_size=5, **kwargs)

    cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser,
                                                                       self_calibration_flag=False, normalize_eigs=True, kernel_size=5, **kwargs)

    p_kwargs = {'vmin': np.nanmin((est_values["p"][..., 0], true_values["p"][..., 0], self_est_values["p"][..., 0], self_true_values["p"][..., 0])),
                'vmax': np.nanmax((true_values["p"][..., 0], true_values["p"][..., 0], self_est_values["p"][..., 0], self_true_values["p"][..., 0]))}
    a_kwargs = {'vmin': np.nanmin((est_values["biref"][..., 1, 1], true_values["biref"][..., 1, 1], self_est_values["biref"][..., 1, 1], self_true_values["biref"][..., 1, 1])),
                'vmax': np.nanmax((est_values["biref"][..., 1, 1], true_values["biref"][..., 1, 1], self_est_values["biref"][..., 1, 1], self_true_values["biref"][..., 1, 1]))}
    b_kwargs = {'vmin': np.nanmin((est_values["biref"][..., 1, 2], true_values["biref"][..., 1, 2], self_est_values["biref"][..., 1, 2], self_true_values["biref"][..., 1, 2])),
                'vmax': np.nanmax((est_values["biref"][..., 1, 2], true_values["biref"][..., 1, 2], self_est_values["biref"][..., 1, 2], self_true_values["biref"][..., 1, 2]))}
    c_kwargs = {'vmin': np.nanmin((est_values["biref"][..., 2, 2], true_values["biref"][..., 2, 2], self_est_values["biref"][..., 2, 2], self_true_values["biref"][..., 2, 2])),
                'vmax': np.nanmax((est_values["biref"][..., 2, 2], true_values["biref"][..., 2, 2], self_est_values["biref"][..., 2, 2], self_true_values["biref"][..., 2, 2]))}
    main_show_cost(parser, cost_itr, est_values, true_values, clbk_itr, name='calib', p_kwargs=p_kwargs, a_kwargs=a_kwargs,
                   b_kwargs=b_kwargs, c_kwargs=c_kwargs)
    main_show_cost(parser, self_cost_itr, self_est_values, self_true_values, self_clbk_itr, name='self_calib', p_kwargs=p_kwargs, a_kwargs=a_kwargs,
                   b_kwargs=b_kwargs, c_kwargs=c_kwargs)

    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
    compare_self_and_calib(self_true_values['p'][:, 0], self_est_values['p'][:, 0], true_values['p'][:, 0], est_values['p'][:, 0],
                           xlabel='$P^{\\rm true}$', ylabel='$\hat{P}$', ax=ax[0, 0], n_points=150)
    compare_self_and_calib(self_true_values['biref'][:, 1, 1], self_est_values['biref'][:, 1, 1], true_values['biref'][:, 1, 1], est_values['biref'][:, 1, 1],
                           xlabel='${\\tt a}^{\\rm true}$', ylabel='$\hat{\\tt a}$', ax=ax[0, 1], n_points=150)
    compare_self_and_calib(self_true_values['biref'][:, 1, 2], self_est_values['biref'][:, 1, 2], true_values['biref'][:, 1, 2], est_values['biref'][:, 1, 2],
                           xlabel='${\\tt b}^{\\rm true}$', ylabel='$\hat{\\tt b}$', ax=ax[1, 0], n_points=150)
    compare_self_and_calib(self_true_values['biref'][:, 2, 2], self_est_values['biref'][:, 2, 2], true_values['biref'][:, 2, 2], est_values['biref'][:, 2, 2],
                           xlabel='${\\tt c}^{\\rm true}$', ylabel='$\hat{\\tt c}$', ax=ax[1, 1], n_points=150)
    plt.tight_layout()
    fig.savefig(f'{outputs_dir}/compare_calib_self_calib.pdf', format='pdf', bbox_inches='tight', transparent="True", pad_inches=0.1)
    plt.show()
    pass


def main_plot_n_obs(n_itr=10, n_rotations_list=None, n_rpt=15, parallel=False, n_core=4, **kwargs):
    if n_rotations_list is None:
        n_rotations_list = np.linspace(10, 30, 10, endpoint=True, dtype=int)
    n_rotations_list_ = np.repeat(n_rotations_list, n_rpt)

    if parallel:
        with mp.Pool(n_core) as p:
            results = p.map(_run_main_plot_n_obs, zip(n_rotations_list_, repeat(n_itr), repeat(kwargs)))
    else:
        results = []
        for n_rotations in n_rotations_list_:
            results.append(_run_main_plot_n_obs((n_rotations, n_itr, dict(kernel_size=5))))
    res_cost_arr = np.array(results).reshape((len(n_rotations_list), n_rpt, 7))

    rot_p_mse, p_err = np.mean(res_cost_arr[..., 1], axis=1), np.std(res_cost_arr[..., 1], axis=1)
    rot_biref_mse, biref_err = np.mean(res_cost_arr[..., 2], axis=1), np.std(res_cost_arr[..., 2], axis=1)
    plot_res_comp_plot(n_rotations_list, rot_p_mse, rot_biref_mse, saveto=f"{outputs_dir}/calib_mse_n_rotations.pdf",
                       xlabel="K", ylim1=(0, None), ylim2=(0, None), p_mse_err=p_err, biref_mse_err=biref_err)
    print(f"ROT p_mse: {rot_p_mse}")
    print(f"ROT biref_mse: {rot_biref_mse}")
    return n_rotations_list, res_cost_arr


def _run_main_plot_n_obs(inputs):
    n_rotations, n_itr, kwargs = inputs

    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"], solar_cut=5 * u.deg)
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())
    cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser,
                                                                       disable=True, normalize_eigs=True,
                                                                       **kwargs)
    mean_num_electrons = np.mean((true_values["images"] / A_gamma).to('').value)
    return (cost_itr[-1] / mean_num_electrons,) + clbk_itr[-1]


def main_plot_exp_time(n_rotations=30, n_rpt=15, n_itr=10, exposure_time_list=None, parallel=False, n_core=4, **kwargs):
    if exposure_time_list is None:
        exposure_time_list = np.logspace(np.log10(5), np.log10(30), 10)
    exposure_time_list_ = np.repeat(exposure_time_list, n_rpt)

    if parallel:
        with mp.Pool(n_core) as p:
            results = p.map(_run_main_plot_exp_time, zip(exposure_time_list_, repeat(n_rotations), repeat(n_itr), repeat(kwargs)))
    else:
        results = []
        for exposure_time in exposure_time_list_:
            results.append(_run_main_plot_exp_time((exposure_time, n_rotations, n_itr, kwargs)))
    res_cost_arr = np.array(results).reshape((len(exposure_time_list), n_rpt, 7))

    ex_p_mse, p_err = np.mean(res_cost_arr[..., 1], axis=1), np.std(res_cost_arr[..., 1], axis=1)
    ex_biref_mse, biref_err = np.mean(res_cost_arr[..., 2], axis=1), np.std(res_cost_arr[..., 2], axis=1)

    plot_res_comp_plot(exposure_time_list, ex_p_mse, ex_biref_mse, saveto=f"{outputs_dir}/calib_mse_exposure_time.pdf",
                       xlabel="$\Delta t \;(s)$", ylim1=(0, None), ylim2=(0, None), p_mse_err=p_err, biref_mse_err=biref_err)
    print(f"EXP p_mse: {ex_p_mse}")
    print(f"EXP biref_mse: {ex_biref_mse}")
    return exposure_time_list, res_cost_arr


def _run_main_plot_exp_time(inputs):
    exposure_time, n_rotations, n_itr, kwargs = inputs

    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"], solar_cut=5 * u.deg)
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())

    zodipol.imager.exposure_time = exposure_time * u.s
    cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser,
                                                                       disable=True, normalize_eigs=True, kernel_size=5,
                                                                       **kwargs)
    mean_num_electrons = np.mean((true_values["images"] / A_gamma).to('').value)
    return (cost_itr[-1] / mean_num_electrons,) + clbk_itr[-1]


def main_plot_uncertainty(n_rotations=10, n_itr=10, n_rpt=7, direction_error_list=None, parallel=False, n_core=4, **kwargs):
    # In the direction estimation study, exposure time need to be long enough to get a significant signal, but we do not
    # omit the stars pixels, so exposure time needs to be short enough to avoid full-well.
    if direction_error_list is None:
        direction_error_list = np.logspace(np.log10(0.0001), np.log10(0.2), 10)
    direction_error_list_ = np.repeat(direction_error_list, n_rpt)
    direction_err_kw = [{"direction_uncertainty": direction_error * u.deg} for direction_error in direction_error_list_]

    if parallel:
        with mp.Pool(n_core) as p:
            results = p.map(_run_main_plot_n_obs, zip(repeat(n_rotations), repeat(n_itr), direction_err_kw))
    else:
        results = []
        for kw in direction_err_kw:
            results.append(_run_main_plot_n_obs((n_rotations, n_itr, kw)))
    res_cost_arr = np.array(results).reshape((len(direction_error_list), n_rpt, 7))

    dir_p_mse, p_err = np.mean(res_cost_arr[..., 1], axis=1), np.std(res_cost_arr[..., 1], axis=1)
    dir_biref_mse, biref_err = np.mean(res_cost_arr[..., 2], axis=1), np.std(res_cost_arr[..., 2], axis=1)

    plot_res_comp_plot(direction_error_list, dir_p_mse, dir_biref_mse, saveto=f"{outputs_dir}/calib_mse_direction_error.pdf",
                       xlabel="Direction Error (deg)", ylim1=(0, None), ylim2=(0, None), p_mse_err=p_err, biref_mse_err=biref_err)
    return direction_error_list, res_cost_arr


def biref_smoothing_study(n_rotations=10, n_itr=10, n_rpt=7, smoothing_list=None, parallel=False, n_core=4, **kwargs):
    if smoothing_list is None:
        smoothing_list = [1, 3, 5, 7, 9]
    smoothing_list_ = np.repeat(smoothing_list, n_rpt)
    biref_smoothing = [dict(kernel_size=smoothing, **kwargs) for smoothing in smoothing_list_]

    if parallel:
        with mp.Pool(n_core) as p:
            results = p.map(_run_main_plot_n_obs, zip(repeat(n_rotations), repeat(n_itr), biref_smoothing))
    else:
        results = []
        for kw in biref_smoothing:
            results.append(_run_main_plot_n_obs((n_rotations, n_itr, kw)))
    res_cost_arr = np.array(results).reshape((len(smoothing_list), n_rpt, 7))

    rot_p_mse, p_err = np.mean(res_cost_arr[..., 1], axis=1), np.std(res_cost_arr[..., 1], axis=1)
    rot_biref_mse, biref_err = np.mean(res_cost_arr[..., 2], axis=1), np.std(res_cost_arr[..., 2], axis=1)
    print('Kernel size MSE for P: ', rot_p_mse)
    print('Kernel size MSE for Biref: ', rot_biref_mse)
    return smoothing_list, res_cost_arr



def main():
    compare_calib_self_calib(n_rotations=30, n_itr=5)
    n_obs_list, cost_n_obs = main_plot_n_obs(n_itr=5, parallel=True, n_core=60)
    exp_t_list, cost_expo = main_plot_exp_time(n_rotations=30, n_itr=5, parallel=True, n_core=60)
    dir_unc_list, cost_dir_unc = main_plot_uncertainty(n_rotations=30, n_itr=5)


if __name__ == '__main__':
    os.mkdir(outputs_dir)
    main()
