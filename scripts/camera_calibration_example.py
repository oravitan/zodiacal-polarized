import os
import pickle as pkl
import logging
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from functools import partial
from tqdm import tqdm

# import the necessary modules
from zodipol.estimation.calibration import Calibration
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def get_observations(n_rotations):
    rotations_file_path = 'saved_models/self_calibration_40.pkl'
    rotation_list = np.linspace(0, 360, n_rotations, endpoint=False)
    if os.path.isfile(rotations_file_path):
        # saved rotations pickle file exists
        with open(rotations_file_path, 'rb') as f:
            obs_rot = pkl.load(f)
        ind_list = np.linspace(0, len(obs_rot), n_rotations, endpoint=False, dtype=int)
        rotation_list = np.linspace(0, 360, len(obs_rot), endpoint=False)[ind_list]
        obs_rot = [obs_rot[ii] for ii in ind_list]
    else:
        obs_rot = [zodipol.create_observation(theta=parser["direction"][0], phi=parser["direction"][1], roll=t * u.deg,
                                              lonlat=False, new_isl=parser["new_isl"]) for t in tqdm(rotation_list)]
        with open(rotations_file_path, 'wb') as f:
            pkl.dump(obs_rot, f)
    return obs_rot, rotation_list

def get_initial_parameters(obs, parser, zodipol):
    motion_blur = 360 / n_rotations / zodipol.imager.exposure_time.value * u.deg
    obs = [o.add_radial_blur(motion_blur, list(parser["resolution"])) for o in obs]
    obs = [o.add_direction_uncertainty(parser["fov"], parser["resolution"], parser["direction_uncertainty"]) for o in obs]

    delta_val, phi_val = np.pi / 4, np.pi / 6
    # delta_val, phi_val = 0, 0
    delta = zodipol.imager.get_birefringence_mat(delta_val, 'center', flat=True, inv=True)
    phi = zodipol.imager.get_birefringence_mat(phi_val, 'linear', flat=True, angle=-np.pi / 4)
    mueller_truth = zodipol.imager.get_birefringence_mueller_matrix(delta, phi)

    obs_biref = [zodipol.imager.apply_birefringence(o, mueller_truth[:, None, ...]) for o in obs]

    polarization_angle = parser["polarization_angle"]
    _, polarization_angle_spatial_diff = np.meshgrid(np.arange(parser["resolution"][0]), np.deg2rad(np.linspace(-10, 10, parser["resolution"][1])), indexing='ij')
    polarization_angle_spatial_diff = np.zeros_like(polarization_angle_spatial_diff)
    polarization_angle_real = polarization_angle[None, None :] + polarization_angle_spatial_diff.flatten()[:, None, None]

    polarizance, _ = np.meshgrid(np.linspace(0.7, 0.95, parser["resolution"][0]), np.arange(parser["resolution"][1]),
                       indexing='ij')
    polarizance_real = polarizance.reshape((-1, 1, 1))

    obs_orig = [zodipol.make_camera_images(o, polarizance_real, polarization_angle_real, n_realizations=parser["n_realizations"], add_noise=True) for o in obs_biref]
    images_orig = zodipol.post_process_images(np.stack(obs_orig, axis=-1))
    return obs, images_orig, polarizance_real, polarization_angle_spatial_diff.reshape((-1)), mueller_truth


def cost_callback(calib: Calibration, p, eta, mueller):
    mueller_est = calib.biref
    p_cost = np.nanmean((p.squeeze() - calib.p)**2)
    mueller_cost = np.nanmean((mueller[..., 1:3, 1:3] - mueller_est[..., 1:3, 1:3])**2)
    return p_cost, mueller_cost


def plot_mueller(mueller, cbar=False, saveto=None):
    mueller = mueller[..., :3, :3]
    fig, ax = plt.subplots(3,3, figsize=(6,6), sharex='col', sharey='row', subplot_kw={'xticks': [], 'yticks': []})
    for i in range(3):
        for j in range(3):
            c = ax[i,j].imshow(mueller[..., i, j].reshape(parser["resolution"]), vmin=mueller.min(), vmax=mueller.max())
            # ax[i,j].get_xaxis().set_visible(False)
            # ax[i,j].get_yaxis().set_visible(False)
    ax[0,0].set_ylabel(0, fontsize=16)
    ax[1,0].set_ylabel(1, fontsize=16)
    ax[2,0].set_ylabel(2, fontsize=16)
    ax[2,0].set_xlabel(0, fontsize=16)
    ax[2,1].set_xlabel(1, fontsize=16)
    ax[2,2].set_xlabel(2, fontsize=16)
    # fig.colorbar(c, ax=ax.ravel().tolist())
    if cbar:
        cb = fig.colorbar(c, ax=ax.ravel().tolist())
        cb.ax.tick_params(labelsize=14)
    else:
        plt.tight_layout(w_pad=-15.0, h_pad=1.0)
    if saveto is not None:
        plt.savefig(saveto, format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()


if __name__ == '__main__':
    # set params
    logging.info(f'Started run.')
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetary"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    n_rotations = 20
    n_itr = 10
    obs, rotation_list = get_observations(n_rotations)

    obs, images_orig, polarizance_real, polarization_angle_real, mueller_truth = get_initial_parameters(obs, parser, zodipol)
    obs_comb = zodipol.combine_observations(obs, polarizance=polarizance_real.squeeze(), polarization_angle=polarization_angle_real)
    callback_partial = partial(cost_callback, p=polarizance_real, eta=polarization_angle_real, mueller=mueller_truth)

    calib = Calibration(obs_comb, zodipol, parser)
    init = {'eta': polarization_angle_real}
    p, eta, biref, cost, itr_cost = calib.calibrate(images_orig, n_itr=n_itr, mode="all", callback=callback_partial, init=init)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 5))
    for ax_i, c in zip(ax, [cost] + list(zip(*itr_cost))):
        ax_i.semilogy(c, lw=3)
        ax_i.grid()
        ax_i.tick_params(labelsize=14)
    ax[-1].set_xlabel("Iteration", fontsize=16)
    ax[0].set_ylabel("Intensity MSE\n(electrons)", fontsize=16)
    ax[1].set_ylabel("$\hat{P}$ MSE", fontsize=16)
    ax[2].set_ylabel("$\hat{B}$ MSE", fontsize=16)
    # plt.suptitle("Cost vs. Iteration", fontsize=18)
    plt.tight_layout()
    plt.savefig("outputs/calib_cost_vs_iteration.pdf", format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()

    # plot comparison of P,eta estimated vs true
    fig, ax = plt.subplots(1, 2, figsize=(5, 3), subplot_kw={'xticks': [], 'yticks': []})
    c1 = ax[0].imshow(polarizance_real.reshape((parser["resolution"][0], parser["resolution"][1])), vmin=0.7, vmax=1)
    ax[0].set_title("$P^{true}$")
    plt.colorbar(c1, ax=ax[0])
    c2 = ax[1].imshow(p.reshape((parser["resolution"][0], parser["resolution"][1])), vmin=0.7, vmax=1)
    ax[1].set_title("$\hat{P}$")
    plt.colorbar(c2, ax=ax[1])
    plt.savefig("outputs/calib_p_eta.pdf", format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()

    plot_mueller(biref, cbar=True, saveto="outputs/calib_mueller_estimation.pdf")
    pass
