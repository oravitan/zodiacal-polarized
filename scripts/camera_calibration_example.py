import logging
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter

# import the necessary modules
from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol, Observation
from zodipol.visualization.skymap_plots import plot_satellite_image, plot_satellite_image_indices

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def make_img(obs, p, add_noise=False, **kwargs):
    if add_noise:
        biref_camera_intensity = zodipol.make_camera_images(obs, p, parser["polarization_angle"], n_realizations=parser["n_realizations"], add_noise=add_noise, **kwargs)
    else:
        biref_camera_intensity = zodipol.make_camera_images(obs, p, parser["polarization_angle"], n_realizations=1, add_noise=add_noise, **kwargs)
    return biref_camera_intensity


if __name__ == '__main__':
    # set params
    logging.info(f'Started run.')
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"], planetary=parser["planetry"], isl=parser["isl"],
                      resolution=parser["resolution"], imager_params=parser["imager_params"])

    n_rotations = 12
    obs = [zodipol.create_observation(theta=parser["direction"][0], phi=parser["direction"][1], roll=t * u.deg, lonlat=False,
                                     new_isl=parser["new_isl"]) for t in np.linspace(0, 360, n_rotations, endpoint=False)]

    # polarizance = np.clip(np.random.normal(parser["polarizance"], 0.03, (len(obs), 1, 1)), a_min=0, a_max=1)
    polarizance, _ = np.meshgrid(np.linspace(0.8, 0.9, parser["resolution"][0]), np.arange(parser["resolution"][1]),
                       indexing='ij')
    polarizance = polarizance.reshape((len(obs[0]), 1, 1))
    obs_orig = [zodipol.make_camera_images(o, polarizance, parser["polarization_angle"],
                                           n_realizations=parser["n_realizations"], add_noise=True) for o in obs]
    images_orig = np.stack(obs_orig, axis=-1)

    # plot_satellite_image_indices(obs_orig, 4, resolution=parser["resolution"])

    p_t = np.linspace(0.6, 1, 20)
    img = [np.stack([zodipol.make_camera_images(o, p, parser["polarization_angle"], n_realizations=1, add_noise=False
                                                ) for p in p_t], axis=-1) for o in obs]
    img_stack = np.stack(img, axis=-2)

    diff_resh = (img_stack - images_orig[..., None]).value.reshape(parser["resolution"] + [parser["n_polarization_ang"], n_rotations, len(p_t)])
    # gg = gaussian_filter(diff_resh, (5, 5, 0, 0), mode='nearest')

    c = np.nansum((1e23 * diff_resh) ** 2, axis=(-3, -2))
    p_est = p_t[np.argmin(np.nan_to_num(c, nan=np.inf), axis=-1)]
    # dd = (delta_m - delta_est) / (delta_m + 1e-10)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    c1 = ax1.imshow(p_est.reshape(parser["resolution"]))
    c1.set_clim(0.8, 0.9)
    plt.colorbar(c1, ax=ax1)
    ax1.set_title('Estimated Per-Pixel\n Polarizance')
    c2 = ax2.imshow(polarizance.reshape(parser["resolution"]))
    c2.set_clim(0.8, 0.9)
    plt.colorbar(c2, ax=ax2)
    ax2.set_title('Actual Per-Pixel\n Polarizance')
    fig.tight_layout()
    plt.savefig('outputs/polarizance_estimation.pdf')
    plt.show()

    pass