import logging
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
from tqdm import tqdm
from PIL import Image

from zodipol.utils.argparser import ArgParser
from zodipol.zodipol.zodipol import Zodipol


logging.basicConfig(level=logging.FATAL)


def generate_video(theta_2d_arr, phi_2d_arr, roll, saveto, rate=10):
    camera_res, vec_res = [], []
    for theta, phi in tqdm(zip(theta_2d_arr.flatten(), phi_2d_arr.flatten()), total=len(theta_2d_arr.flatten())):
        obs = zodipol.create_observation(theta=theta, phi=phi, roll=roll, lonlat=False, new_isl=parser["new_isl"])
        camera_intensity = zodipol.make_camera_images_multicolor(obs, n_realizations=1, add_noise=False)
        camera_res.append(camera_intensity)

    camera_res_arr = np.stack(camera_res, axis=-1)
    camera_res_chs = np.log10(camera_res_arr[:, 0, 0, :].value)
    camera_res_norm = (camera_res_chs - camera_res_chs.min()) / (camera_res_chs.max() - camera_res_chs.min())

    frames = []  # for storing the generated images
    for ii in range(camera_res_norm.shape[-1]):
        fig = plt.figure()
        plt.imshow(camera_res_norm[:, ii].reshape(parser['resolution']), vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout(pad=0)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        frames.append(frame)
    output_data = np.stack(frames, axis=0).astype(np.uint8)
    skvideo.io.vwrite(saveto, output_data, inputdict={'-r': str(rate)}, outputdict={'-r': str(rate)})
    pass


if __name__ == '__main__':
    logging.info(f'Started run.')
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"],
                      n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"],
                      n_freq=parser["n_freq"],
                      planetary=parser["planetary"], isl=parser["isl"], resolution=parser["resolution"],
                      imager_params=parser["imager_params"], solar_cut=5 * u.deg)
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())
    
    theta_arr = np.arange(120, 60, -0.5) * u.deg
    phi_arr = np.linspace(-100, -80, len(theta_arr)) * u.deg
    generate_video(theta_arr, phi_arr, -90*u.deg, "outputs/milky_way_movie.mp4", rate=20)
    
    theta_arr = [90] * u.deg
    phi_arr = np.arange(50, 110 - 360, -1) * u.deg
    phi_2d_arr, theta_2d_arr = np.meshgrid(phi_arr, theta_arr)
    generate_video(theta_2d_arr, phi_2d_arr, -90*u.deg, "outputs/camera_movie.mp4", rate=20)

