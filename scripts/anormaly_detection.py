"""
This script performs self-calibration on a Zodipol object.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the necessary modules
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol, get_observations, get_initial_parameters
from zodipol.visualization.calibration_plots import plot_deviation_comp, plot_mueller, plot_cost_itr
from scripts.self_calibration import perform_estimation


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
    obs_truth, images_res, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(obs_truth, parser, zodipol, mode='anomalies')
    images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
    images_res_flat = zodipol.post_process_images(images_res_flat)
    cost_itr, est_values, clbk_itr = perform_estimation(zodipol, parser, rotation_list, images_res_flat,
                                                  polarizance_real, polarization_angle_real, mueller_truth, n_itr=n_itr)

    E = est_values['p'] - 0.9
    sigma = np.nanstd(E)
    z_k = np.abs(E[:, 0] / sigma)

    gt = polarizance_real[:, 0] <= 0.5
    gt[np.isnan(polarizance_real[:, 0])] = np.nan

    pd.DataFrame(dict(value=z_k, gt=gt)).dropna().boxplot(column='value', by='gt')
    plt.title(None)
    plt.suptitle(None)
    plt.xlabel('Affected pixels', fontsize=16)
    plt.ylabel('Z-score', fontsize=16)
    plt.gca().tick_params(labelsize=14)
    plt.savefig('outputs/anomality_boxplot.pdf', format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()
    pass


if __name__ == '__main__':
    main()
