import logging
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from functools import partial

# import the necessary modules
from zodipol.estimation.calibration import Calibration
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol, get_observations, get_initial_parameters
from zodipol.visualization.calibration_plots import plot_mueller, plot_cost_itr, plot_deviation_comp, plot_res_comp_plot

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def cost_callback(calib: Calibration, p, eta, mueller):
    mueller_est = calib.biref
    p_cost = np.nanmean((p.squeeze() - calib.p)**2)
    mueller_cost = np.nanmean((mueller[..., 1:3, 1:3] - mueller_est[..., 1:3, 1:3])**2)
    return p_cost, mueller_cost


if __name__ == '__main__':
    # set params
    logging.info(f'Started run.')
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    n_rotations = 20
    n_itr = 20
    obs, rotation_list = get_observations(n_rotations, zodipol, parser)

    obs, images_orig, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(obs, parser, zodipol)
    obs_comb = zodipol.combine_observations(obs, polarizance=polarizance_real.squeeze(), polarization_angle=polarization_angle_real)
    callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real, mueller=mueller_truth)

    calib = Calibration(obs_comb, zodipol, parser)
    init = {'eta': polarization_angle_real}
    p, eta, biref, cost, itr_cost = calib.calibrate(images_orig, n_itr=n_itr, callback=callback_partial, init=init, normalize_eigs=True)
    p_cost, mueller_cost = list(zip(*itr_cost))

    plot_cost_itr(cost, p_cost, mueller_cost, saveto="outputs/calib_cost_vs_iteration.pdf")
    plot_deviation_comp(parser, polarizance_real[:, 0], p[:, 0], saveto="outputs/calib_mueller_estimation.pdf")

    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())

    # calibration vs. number of observations
    n_rotations_list = [4, 6, 10, 14, 18, 22, 26, 30]
    res_cost = []
    for n_rotations in n_rotations_list:
        obs, rotation_list = get_observations(n_rotations, zodipol, parser)
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
    plot_res_comp_plot(n_rotations_list, rot_p_mse, rot_biref_mse, saveto="outputs/calib_mse_n_rotations.pdf", xlabel="Number of observations")

    # now estimate how well we calibration based on exposure time
    n_rotations = 20
    n_itr = 10
    exposure_time_list = np.logspace(np.log10(0.5), np.log10(100), 20)
    res_cost = []
    for exposure_time in exposure_time_list:
        zodipol.imager.exposure_time = exposure_time * u.s
        obs, rotation_list = get_observations(n_rotations, zodipol, parser)
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
    plot_res_comp_plot(exposure_time_list, ex_p_mse, ex_biref_mse, saveto="outputs/calib_mse_exposure_time.pdf", xlabel="$\Delta t \;(s)$")

