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
from zodipol.visualization.calibration_plots import plot_deviation_comp, plot_mueller, plot_cost_itr


def perform_estimation(zodipol, parser, rotation_list, images_res_flat, polarizance_real, polarization_angle_real, mueller_truth, n_itr=10):
    theta0, phi0 = zodipol.create_sky_coords(theta=parser["direction"][0], phi=parser["direction"][1], roll=0 * u.deg, resolution=parser["resolution"])
    callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real, mueller=mueller_truth)
    self_calib = SelfCalibration(images_res_flat, rotation_list, zodipol, parser, theta=theta0, phi=phi0)
    init_dict = {'eta': polarization_angle_real}
    _, _, _, cost_itr, clbk_itr = self_calib.calibrate(images_res_flat, n_itr=n_itr, callback=callback_partial, init=init_dict)
    p, eta, biref = self_calib.get_properties()
    return cost_itr, p, eta, biref, clbk_itr


def cost_callback(calib: SelfCalibration, p, eta, mueller):
    cp, _, biref = calib.get_properties()
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
    n_itr = 20
    n_rotations = 20
    obs_truth, rotation_list = get_observations(n_rotations, zodipol, parser)
    obs_truth, images_res, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(obs_truth, parser, zodipol)
    images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
    images_res_flat = zodipol.post_process_images(images_res_flat)
    cost_itr, p_hat, eta_hat, biref, clbk_itr = perform_estimation(zodipol, parser, rotation_list, images_res_flat,
                                                  polarizance_real, polarization_angle_real, mueller_truth, n_itr=n_itr)
    p_cost, mueller_cost = list(zip(*clbk_itr))
    plot_cost_itr(cost_itr, p_cost, mueller_cost, saveto='outputs/self_calibration_cost_itr.pdf')
    plot_deviation_comp(parser, polarizance_real[..., 0], p_hat[..., 0], set_colors=True, saveto='outputs/self_calibration_polarizance_est.pdf')
    plot_mueller(biref, parser, cbar=True, vmin=-0.05, vmax=1, saveto='outputs/self_calib_birefringence_est.pdf')
    pass


if __name__ == '__main__':
    main()
