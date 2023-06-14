"""
This script performs self-calibration on a Zodipol object.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import medfilt2d
from sklearn.metrics import confusion_matrix, roc_curve

# import the necessary modules
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol, get_observations, get_initial_parameters, get_initialization
from zodipol.visualization.calibration_plots import plot_deviation_comp, plot_mueller, plot_cost_itr
from scripts.self_calibration import self_calibrate


def main():
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    # generate observations
    n_itr = 5
    n_rotations = 30
    obs_truth, rotation_list = get_observations(n_rotations, zodipol, parser)
    obs_truth, images_res, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(obs_truth, parser, zodipol, mode='anomalies')
    images_res_flat = images_res.reshape((np.prod(parser["resolution"]), parser["n_polarization_ang"], n_rotations))
    images_res_flat = zodipol.post_process_images(images_res_flat)
    initialization = get_initialization(polarizance_real, mueller_truth)
    star_pixels = np.stack([o.star_pixels for o in obs_truth], axis=-1)
    cost_itr, est_values, clbk_itr = self_calibrate(zodipol, parser, rotation_list, images_res_flat,
                                                    polarizance_real, polarization_angle_real, mueller_truth,
                                                    n_itr=n_itr, max_p=np.max(polarizance_real),
                                                    initialization=initialization, star_pixels=star_pixels)

    # Filtering Method
    p_est = est_values['p'][:, 0].reshape((300, 200))
    p_filt = medfilt2d(p_est, 5)
    p_diff = p_est - p_filt
    p_diff_w = (p_diff.ravel() - np.nanmean(p_diff)) / np.nanstd(p_diff)

    res_ind = ~np.isnan(p_diff_w) & ~np.isnan(polarizance_real[:, 0])

    classification_score = abs(p_diff_w.ravel()[res_ind])
    classification = classification_score >= 2.5
    classification = classification
    classification = np.where(classification, 'damaged', 'non-damaged')

    gt = polarizance_real[:, 0] <= (polarizance_real.max() + polarizance_real.min()) / 2
    gt = gt[res_ind]
    gt = np.where(gt, 'damaged', 'non-damaged')

    pd.DataFrame(dict(value=classification_score, gt=gt)).dropna().boxplot(column='value', by='gt')
    plt.title(None)
    plt.suptitle(None)
    plt.xlabel('Affected pixels', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.gca().tick_params(labelsize=14)
    plt.savefig('outputs/anomality_boxplot.pdf', format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()

    conf_matrix = confusion_matrix(gt, classification)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    ax.tick_params(labelsize=16)
    ax.axis('auto')
    plt.xticks([0, 1], ['damaged', 'non-damaged'], fontsize=18)
    plt.yticks([0, 1], ['damaged', 'non-damaged'], fontsize=18, va='center')
    plt.xlabel('Estimations', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.tight_layout()
    plt.savefig('outputs/anomality_confusion.pdf', format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()

    # Density-based Method
    density_score = p_est.ravel()[res_ind]
    density_score_white = (density_score - np.mean(density_score)) / np.std(density_score)
    classification_score = abs(density_score_white)
    classification = classification_score >= 2.5
    classification = classification
    classification = np.where(classification, 'damaged', 'non-damaged')

    gt = polarizance_real[:, 0] <= (polarizance_real.max() + polarizance_real.min()) / 2
    gt = gt[res_ind]
    gt = np.where(gt, 'damaged', 'non-damaged')

    pd.DataFrame(dict(value=classification_score, gt=gt)).dropna().boxplot(column='value', by='gt')
    plt.title(None)
    plt.suptitle(None)
    plt.xlabel('Affected pixels', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.gca().tick_params(labelsize=14)
    plt.savefig('outputs/anomality_boxplot.pdf', format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()

    conf_matrix = confusion_matrix(gt, classification)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    ax.tick_params(labelsize=16)
    ax.axis('auto')
    plt.xticks([0, 1], ['damaged', 'non-damaged'], fontsize=18)
    plt.yticks([0, 1], ['damaged', 'non-damaged'], fontsize=18, va='center')
    plt.xlabel('Estimations', fontsize=18)
    plt.ylabel('Actuals', fontsize=18, rotation=0)
    plt.title('Confusion Matrix', fontsize=18)
    plt.tight_layout()
    plt.savefig('outputs/anomality_confusion.pdf', format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()

    pass


if __name__ == '__main__':
    main()
