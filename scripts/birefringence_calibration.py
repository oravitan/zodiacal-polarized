import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# import the necessary modules
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol, Observation
from zodipol.visualization.skymap_plots import plot_satellite_image, plot_satellite_image_indices


def make_biref_img(obs, delta, phi, add_noise=False, n_realizations=1, **kwargs):
    if not isinstance(delta, np.ndarray) and not isinstance(delta, np.ndarray):
        delta, phi = np.repeat(delta, len(obs)), np.repeat(phi, len(obs))
    biref_mueller = zodipol.imager.get_birefringence_mueller_matrix(delta, phi)
    biref_obs = zodipol.imager.apply_birefringence(obs, biref_mueller)
    if add_noise:
        biref_camera_intensity = zodipol.make_camera_images(biref_obs, parser["polarizance"], parser["polarization_angle"], n_realizations=n_realizations, add_noise=add_noise, **kwargs)
    else:
        biref_camera_intensity = zodipol.make_camera_images(biref_obs, parser["polarizance"], parser["polarization_angle"], n_realizations=1, add_noise=add_noise, **kwargs)
    return biref_camera_intensity


if __name__ == '__main__':
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"], n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"], n_freq=parser["n_freq"], planetary=parser["planetry"], isl=parser["isl"], resolution=parser["resolution"], imager_params=parser["imager_params"])

    # Create observations
    n_rotations = 6
    obs = [zodipol.create_observation(theta=parser["direction"][0], phi=parser["direction"][1], roll=t * u.deg,
                                      lonlat=False, new_isl=parser["new_isl"]) for t in
           tqdm(np.linspace(0, 360, n_rotations, endpoint=False))]

    # Create birefringence images
    delta_val, phi_val = np.pi / 2, np.pi / 2
    delta = zodipol.imager.get_birefringence_mat(delta_val, 'linear', flat=True, angle=np.pi / 4)
    phi = zodipol.imager.get_birefringence_mat(phi_val, 'linear', flat=True, angle=-np.pi / 4)
    biref_camera_intensity = [
        make_biref_img(o, delta, phi, n_realizations=parser["n_realizations"], fillna=0, add_noise=True) for o in
        tqdm(obs)]
    biref_camera_intensity_stack = np.stack(biref_camera_intensity, axis=-1)

    # Plot the birefringence image
    plot_satellite_image_indices(biref_camera_intensity_stack[..., 4], 4, resolution=parser["resolution"])

    # perform calibration grid search
    d_t = np.linspace(-np.pi / 2, np.pi / 2, 12)
    theta_t = np.linspace(-np.pi / 2, np.pi / 2, 13)
    img = [[[make_biref_img(o, d, t, n_realizations=1, add_noise=False) for d in d_t] for t in theta_t] for o in tqdm(obs)]
    img_stack = np.stack(img, axis=-1)
    img_stack_sh = np.moveaxis(img_stack, [0, 1], [-1, -2])

    diff_resh = (img_stack_sh - biref_camera_intensity_stack[..., None, None].value).reshape(
        tuple(parser["resolution"]) + img_stack_sh.shape[1:])
    c = np.nansum((1e21 * diff_resh) ** 2, axis=(2, 3))

    # plot example for the non-convex problem
    plt.figure()
    plt.imshow(c[150, 100, ...], extent=[-np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2])
    plt.xlabel('$\delta$')
    plt.ylabel('$\\theta_{\delta}$')
    plt.colorbar()
    plt.show()

    DT, TT = np.meshgrid(d_t, theta_t)
    DT, TT = DT.flatten(), TT.flatten()
    c_flat = c.reshape(tuple(parser["resolution"]) + (len(DT),))
    min_c = np.argmin(np.nan_to_num(c_flat, nan=np.inf), axis=-1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    c1 = ax1.imshow(DT[min_c])
    c2 = ax2.imshow(TT[min_c])
    c3 = ax3.imshow(delta.reshape(parser["resolution"]))
    c4 = ax4.imshow(phi.reshape(parser["resolution"]))
    plt.colorbar(c1, ax=ax1)
    plt.colorbar(c2, ax=ax2)
    plt.colorbar(c3, ax=ax3)
    plt.colorbar(c4, ax=ax4)
    plt.show()

    pass
