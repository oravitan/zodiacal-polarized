"""
Example script to perform the camera calibration.
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from datetime import datetime
from functools import partial
from tqdm import tqdm

# import the necessary modules
from zodipol.estimation.calibration import Calibration
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol.zodipol import Zodipol
from zodipol.zodipol.generate import get_observations, get_initial_parameters, get_initialization
from zodipol.visualization.calibration_plots import plot_mueller, plot_cost_itr, plot_deviation_comp, plot_res_comp_plot
from scripts.self_calibration import cost_callback

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)

run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
outputs_dir = f'outputs/{run_time}'


def calibration(n_rotations: int, n_itr: int, zodipol: Zodipol, parser: ArgParser, disable=False):
    """
    Function to perform the calibration.
    """
    obs, rotation_list = get_observations(n_rotations, zodipol, parser)
    obs, images_orig, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(obs, parser, zodipol, mode='sine')
    true_values = {'images': images_orig, 'p': polarizance_real, 'eta': polarization_angle_real, 'biref': mueller_truth}
    initialization = get_initialization(polarizance_real, mueller_truth)
    obs_comb = zodipol.combine_observations(obs, polarizance=polarizance_real.squeeze(), polarization_angle=polarization_angle_real)
    callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real, mueller=mueller_truth)

    calib = Calibration(obs_comb, zodipol, parser)
    init = {'eta': polarization_angle_real, **initialization}
    p, eta, biref, cost, itr_cost = calib.calibrate(images_orig, n_itr=n_itr, callback=callback_partial, init=init, disable=disable,
                                                    normalize_eigs=True, kernel_size=9)  # , normalize_eigs=True
    est_images = np.stack([calib.forward_model(o) for o in obs_comb], axis=-1)
    p_cost, mueller_cost, p_std, mueller_std, p_mad, mueller_mad = list(zip(*itr_cost))
    est_values = {'p': p, 'eta': eta, 'biref': biref, 'images': est_images}
    return true_values, est_values, cost, p_cost, mueller_cost


def visualize_calibration(n_rotations=30, n_itr=10):
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    true_values, est_values, cost, p_cost, mueller_cost = calibration(n_rotations, n_itr, zodipol, parser)
    plot_cost_itr(cost, p_cost, mueller_cost, saveto=f"{outputs_dir}/calib_cost_vs_iteration.pdf")
    plot_deviation_comp(parser, true_values["p"][:, 0], est_values["p"][:, 0], saveto=f"{outputs_dir}/calib_p_estimation.pdf")
    plot_mueller(est_values["biref"] - np.eye(3)[None, ...], parser, cbar=True, saveto=f'{outputs_dir}/calibration_biref_matrix_reconst.pdf')
    plot_mueller(true_values["biref"][..., :3, :3] - np.eye(3)[None, ...], parser, cbar=True, saveto=f'{outputs_dir}/mueller_matrix_example.pdf')
    pass


def plot_calibration_nbos(n_itr=10, n_rotations_list=None):
    if n_rotations_list is None:
        n_rotations_list = np.linspace(10, 30, 10, endpoint=True, dtype=int)
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())

    # calibration vs. number of observations
    res_cost = []
    for n_rotations in tqdm(n_rotations_list):
        true_values, est_values, cost, p_cost, mueller_cost = calibration(n_rotations, n_itr, zodipol, parser, disable=True)
        mean_num_electrons = np.mean((true_values["images"] / A_gamma).to('').value)
        res_cost.append((cost[-1] / mean_num_electrons, p_cost[-1], mueller_cost[-1]))
    rot_intensity_mse, rot_p_mse, rot_biref_mse = list(zip(*res_cost))
    plot_res_comp_plot(n_rotations_list, rot_p_mse, rot_biref_mse, saveto=f"{outputs_dir}/calib_mse_n_rotations.pdf",
                       xlabel="Number of observations")


def plot_calibration_exp(n_rotations=30, n_itr=10, exposure_time_list=None):
    if exposure_time_list is None:
        exposure_time_list = np.logspace(np.log10(10), np.log10(60), 10)
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())
    res_cost = []
    for exposure_time in tqdm(exposure_time_list):
        zodipol.imager.exposure_time = exposure_time * u.s
        true_values, est_values, cost, p_cost, mueller_cost = calibration(n_rotations, n_itr, zodipol, parser, disable=True)
        mean_num_electrons = np.mean((true_values['images'] / A_gamma).to('').value)
        res_cost.append((cost[-1] / mean_num_electrons, p_cost[-1], mueller_cost[-1]))
    ex_intensity_mse, ex_p_mse, ex_biref_mse = list(zip(*res_cost))
    plot_res_comp_plot(exposure_time_list, ex_p_mse, ex_biref_mse, saveto=f"{outputs_dir}/calib_mse_exposure_time.pdf",
                       xlabel="$\Delta t \;(s)$")


if __name__ == '__main__':
    os.mkdir(outputs_dir)
    # set params
    logging.info(f'Started run.')
    visualize_calibration(n_rotations=30, n_itr=10)
    plot_calibration_nbos(n_itr=30, n_rotations_list=None)
    plot_calibration_exp(n_rotations=30, n_itr=30, exposure_time_list=None)
    pass
