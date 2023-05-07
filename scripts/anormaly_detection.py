"""
This script performs self-calibration on a Zodipol object.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve

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
    gt = np.where(gt, 'damaged', 'non-damaged')

    pd.DataFrame(dict(value=z_k, gt=gt)).dropna().boxplot(column='value', by='gt')
    plt.title(None)
    plt.suptitle(None)
    plt.xlabel('Affected pixels', fontsize=16)
    plt.ylabel('Z-score', fontsize=16)
    plt.gca().tick_params(labelsize=14)
    plt.savefig('outputs/anomality_boxplot.pdf', format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()

    classficiation = z_k >= 3
    classficiation = np.where(classficiation, 'damaged', 'non-damaged')

    conf_matrix = confusion_matrix(gt, classficiation)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    ax.tick_params(labelsize=16)
    plt.xticks([0, 1], ['damaged', 'non-damaged'], fontsize=18)
    plt.yticks([0, 1], ['damaged', 'non-damaged'], fontsize=18, rotation=90, va='center')
    plt.xlabel('Estimations', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig('outputs/anomality_confusion.pdf', format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
