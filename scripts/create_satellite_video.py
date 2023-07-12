import logging
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import healpy as hp
from tqdm import tqdm
from PIL import Image

from zodipol.utils.argparser import ArgParser
from zodipol.zodipol.zodipol import Zodipol
from zodipol.zodipol.observation import Observation
from zodipol.visualization.skymap_plots import plot_satellite_image, plot_satellite_image_indices, plot_skymap

logging.basicConfig(level=logging.FATAL)


if __name__ == '__main__':
    logging.info(f'Started run.')
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"],
                      planetary=parser["planetary"], isl=parser["isl"], resolution=parser["resolution"],
                      imager_params=parser["imager_params"], solar_cut=5 * u.deg)
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())

    obs_full = zodipol.create_full_sky_observation(nside=128, obs_time=parser["obs_time"])
    camera_intensity_full_color = zodipol.make_camera_images_multicolor(obs_full, n_realizations=parser["n_realizations"], add_noise=False)


    theta_arr = [90] * u.deg
    phi_arr = np.arange(65, 75 - 360, -1) * u.deg
    phi_2d_arr, theta_2d_arr = np.meshgrid(phi_arr, theta_arr)
    camera_res, vec_res = [], []
    for theta, phi in tqdm(zip(theta_2d_arr.flatten(), phi_2d_arr.flatten()), total=len(theta_2d_arr.flatten())):
        obs = zodipol.create_observation(theta=theta, phi=phi, roll=-90*u.deg, lonlat=False, new_isl=parser["new_isl"])
        camera_intensity = zodipol.make_camera_images_multicolor(obs, n_realizations=1, add_noise=False)
        camera_res.append(camera_intensity)

    camera_res_arr = np.stack(camera_res, axis=-1)
    camera_res_chs = camera_res_arr[:, 0, 0, :]
    camera_res_norm = (camera_res_chs - camera_res_chs.min()) / (camera_res_chs.max() - camera_res_chs.min())

    frames = []  # for storing the generated images
    for ii in range(camera_res_norm.shape[-1]):
        fig = plt.figure()
        plt.imshow(camera_res_norm[:, ii].reshape(parser['resolution']), vmin=0, vmax=1)
        plt.axis('off')
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        frames.append(frame)

    imgs = [Image.fromarray(img) for img in frames]
    imgs[0].save("outputs/camera_movie.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
