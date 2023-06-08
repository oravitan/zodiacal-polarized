"""
This script performs self-calibration on a Zodipol object.
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from tqdm import tqdm
from functools import partial
from typing import List

# import the necessary modules
from zodipol.estimation.self_calibration import SelfCalibration
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol, get_observations, get_initial_parameters, get_initialization
from zodipol.visualization.calibration_plots import plot_deviation_comp, plot_mueller, plot_cost_itr, plot_res_comp_plot


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


def run_self_calibration(n_rotations, n_itr, zodipol, parser, disable=False, direction_uncertainty=None, **kwargs):
    obs_truth, rotation_list = get_observations(n_rotations, zodipol, parser)
    obs_truth, images_res, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(
        obs_truth, parser, zodipol, mode='sine', direction_uncertainty=direction_uncertainty)
    images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
    images_res_flat = zodipol.post_process_images(images_res_flat)
    initialization = get_initialization(polarizance_real, mueller_truth)
    star_pixels = np.stack([o.star_pixels for o in obs_truth], axis=-1)
    cost_itr, est_values, clbk_itr = self_calibrate(zodipol, parser, rotation_list, images_res_flat,
                                                    polarizance_real, polarization_angle_real,
                                                    mueller_truth, n_itr=n_itr, disable=disable, max_p=np.max(polarizance_real),
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


def main_show_cost(n_rotations=10, n_itr=10):
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    # generate observations
    cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser, normalize_eigs=True)
    p_cost, mueller_cost, p_std, mueller_std, p_mad, mueller_mad = list(zip(*clbk_itr))
    plot_cost_itr(cost_itr, p_cost, mueller_cost, saveto='outputs/self_calibration_cost_itr.pdf')
    plot_deviation_comp(parser, true_values["p"][..., 0], est_values['p'][..., 0], set_colors=True,
                        saveto='outputs/self_calibration_polarizance_est.pdf')
    plot_mueller(est_values['biref'], parser, cbar=True, saveto='outputs/self_calib_birefringence_est.pdf')
    pass


def main_plot_n_obs(n_itr=10, n_rotations_list=None):
    if n_rotations_list is None:
        n_rotations_list = [4, 6, 10, 14, 18, 22, 26, 30]
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    # generate observations
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())
    res_cost = []
    for n_rotations in tqdm(n_rotations_list):
        cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser,
                                                                           disable=True, normalize_eigs=True)
        mean_num_electrons = np.mean((true_values["images"] / A_gamma).to('').value)
        res_cost.append((cost_itr[-1] / mean_num_electrons,) + clbk_itr[-1])
    rot_intensity_mse, rot_p_mse, rot_biref_mse, rot_p_std, rot_biref_std, rot_p_mad, rot_biref_mad = list(
        zip(*res_cost))
    plot_res_comp_plot(n_rotations_list, rot_p_mse, rot_biref_mse, saveto="outputs/calib_mse_n_rotations.pdf",
                       xlabel="K", ylim1=(0, None), ylim2=(0, None))
    return res_cost


def main_plot_exp_time(n_rotations=10, n_itr=10, exposure_time_list=None):
    if exposure_time_list is None:
        exposure_time_list = np.logspace(np.log10(2), np.log10(50), 10)
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())

    # now estimate how well we calibration based on exposure time
    res_cost = []
    for exposure_time in tqdm(exposure_time_list):
        zodipol.imager.exposure_time = exposure_time * u.s
        cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser,
                                                                           disable=True, normalize_eigs=True)
        mean_num_electrons = np.mean((true_values["images"] / A_gamma).to('').value)
        res_cost.append((cost_itr[-1] / mean_num_electrons,) + clbk_itr[-1])
    ex_intensity_mse, ex_p_mse, ex_biref_mse, ex_p_std, ex_biref_std, ex_p_mad, ex_biref_mad = list(zip(*res_cost))
    plot_res_comp_plot(exposure_time_list, ex_p_mse, ex_biref_mse, saveto="outputs/calib_mse_exposure_time.pdf",
                       xlabel="$\Delta t \;(s)$", ylim1=(0, None), ylim2=(0, None))
    return res_cost


def main_plot_uncertainty(n_rotations=10, n_itr=10, direction_error_list=None):
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
        cost_itr, est_values, true_values, clbk_itr = run_self_calibration(n_rotations, n_itr, zodipol, parser,
                                                                           disable=True, normalize_eigs=True,
                                                                           direction_uncertainty=direction_error * u.deg)
        mean_num_electrons = np.mean((true_values["images"] / A_gamma).to('').value)
        res_cost.append((cost_itr[-1] / mean_num_electrons,) + clbk_itr[-1])
    rot_intensity_mse, dir_p_mse, dir_biref_mse, dir_p_std, dir_biref_std, dir_p_mad, dir_biref_mad = list(
        zip(*res_cost))
    plot_res_comp_plot(direction_error_list, dir_p_mse, dir_biref_mse, saveto="outputs/calib_mse_direction_error.pdf",
                       xlabel="Direction Error (deg)")
    return res_cost


def main():
    main_show_cost(n_rotations=30, n_itr=10)
    cost_n_obs = main_plot_n_obs(n_itr=20)
    cost_expo = main_plot_exp_time(n_itr=20)
    cost_dir_unc = main_plot_uncertainty(n_itr=10)
    pass


if __name__ == '__main__':
    main()
