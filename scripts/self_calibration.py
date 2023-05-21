"""
This script performs self-calibration on a Zodipol object.
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from functools import partial

# import the necessary modules
from zodipol.estimation.self_calibration import SelfCalibration
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol, get_observations, get_initial_parameters
from zodipol.visualization.calibration_plots import plot_deviation_comp, plot_mueller, plot_cost_itr, plot_res_comp_plot


def perform_estimation(zodipol, parser, rotation_list, images_res_flat, polarizance_real, polarization_angle_real, mueller_truth, n_itr=10, **kwargs):
    theta0, phi0 = zodipol.create_sky_coords(theta=parser["direction"][0], phi=parser["direction"][1], roll=0 * u.deg, resolution=parser["resolution"])
    callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real, mueller=mueller_truth)
    self_calib = SelfCalibration(images_res_flat, rotation_list, zodipol, parser, theta=theta0, phi=phi0)
    init_dict = {'eta': polarization_angle_real}
    _, _, _, cost_itr, clbk_itr = self_calib.calibrate(images_res_flat, n_itr=n_itr, callback=callback_partial, init=init_dict, **kwargs)
    images, p, eta, biref = self_calib.get_properties()
    est_values = {'images': images, 'p': p, 'eta': eta, 'biref': biref}
    return cost_itr, est_values, clbk_itr


def cost_callback(calib: SelfCalibration, p, eta, mueller):
    _, cp, _, biref = calib.get_properties()
    p_cost = np.nanmean((p - cp) ** 2)
    mueller_cost = np.nanmean((mueller[..., 1:3, 1:3] - biref[..., 1:3, 1:3])**2)
    return p_cost, mueller_cost


def main():
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    # generate observations
    n_itr = 10
    n_rotations = 20
    obs_truth, rotation_list = get_observations(n_rotations, zodipol, parser)
    obs_truth, images_res, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(obs_truth, parser, zodipol, mode='sine')
    images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
    images_res_flat = zodipol.post_process_images(images_res_flat)
    cost_itr, est_values, clbk_itr = perform_estimation(zodipol, parser, rotation_list, images_res_flat,
                                                  polarizance_real, polarization_angle_real, mueller_truth, n_itr=n_itr, normalize_eigs=True)
    p_cost, mueller_cost = list(zip(*clbk_itr))
    plot_cost_itr(cost_itr, p_cost, mueller_cost, saveto='outputs/self_calibration_cost_itr.pdf')
    plot_deviation_comp(parser, polarizance_real[..., 0], est_values['p'][..., 0], set_colors=True, saveto='outputs/self_calibration_polarizance_est.pdf')
    plot_mueller(est_values['biref'], parser, cbar=True, saveto='outputs/self_calib_birefringence_est.pdf')

    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())
    n_rotations_list = [4, 6, 10, 14, 18, 22, 26, 30]
    res_cost = []
    for n_rotations in n_rotations_list:
        obs_truth, rotation_list = get_observations(n_rotations, zodipol, parser)
        obs_truth, images_res, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(
            obs_truth, parser, zodipol, mode='sine')
        images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
        images_res_flat = zodipol.post_process_images(images_res_flat)
        cost_itr, est_values, clbk_itr = perform_estimation(zodipol, parser, rotation_list, images_res_flat,
                                                                       polarizance_real, polarization_angle_real,
                                                                       mueller_truth, n_itr=n_itr)
        p_cost, mueller_cost = list(zip(*clbk_itr))
        mean_num_electrons = np.mean((images_res_flat / A_gamma).to('').value)
        res_cost.append((cost_itr[-1] / mean_num_electrons, p_cost[-1], mueller_cost[-1]))
    rot_intensity_mse, rot_p_mse, rot_biref_mse = list(zip(*res_cost))
    plot_res_comp_plot(n_rotations_list, rot_p_mse, rot_biref_mse, saveto="outputs/calib_mse_n_rotations.pdf",
                       xlabel="Number of observations")

    # now estimate how well we calibration based on exposure time
    n_rotations = 20
    n_itr = 10
    exposure_time_list = np.logspace(np.log10(0.5), np.log10(100), 20)
    res_cost = []
    for exposure_time in exposure_time_list:
        zodipol.imager.exposure_time = exposure_time * u.s
        obs_truth, rotation_list = get_observations(n_rotations, zodipol, parser)
        obs_truth, images_res, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(
            obs_truth, parser, zodipol, mode='sine')
        images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
        images_res_flat = zodipol.post_process_images(images_res_flat)
        cost_itr, est_values, clbk_itr = perform_estimation(zodipol, parser, rotation_list, images_res_flat,
                                                            polarizance_real, polarization_angle_real,
                                                            mueller_truth, n_itr=n_itr)
        p_cost, mueller_cost = list(zip(*clbk_itr))
        mean_num_electrons = np.mean((images_res_flat / A_gamma).to('').value)
        res_cost.append((cost_itr[-1] / mean_num_electrons, p_cost[-1], mueller_cost[-1]))
    ex_intensity_mse, ex_p_mse, ex_biref_mse = list(zip(*res_cost))
    plot_res_comp_plot(exposure_time_list, ex_p_mse, ex_biref_mse, saveto="outputs/calib_mse_exposure_time.pdf",
                       xlabel="$\Delta t \;(s)$")


if __name__ == '__main__':
    main()
