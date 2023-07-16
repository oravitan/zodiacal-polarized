import logging
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io

from zodipol.utils.argparser import ArgParser
from zodipol.zodipol.zodipol import Zodipol
from zodipol.zodipol.observation import Observation
from zodipol.visualization.skymap_plots import plot_satellite_image_indices

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def generate_video(camera_res_arr, saveto, rate=10):
    camera_res_chs = np.log10(camera_res_arr.value)
    camera_res_norm = (camera_res_chs - camera_res_chs.min()) / (camera_res_chs.max() - camera_res_chs.min())

    frames = []  # for storing the generated images
    for ii in range(camera_res_norm.shape[-1]):
        fig = plt.figure()
        plt.imshow(camera_res_norm[:, ii].reshape(parser['resolution']), vmin=0, vmax=1, cmap='gray')
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
    n_polarization_ang = 30
    polarization_angle = np.linspace(0, np.pi, n_polarization_ang, endpoint=False)
    polarizance = 1

    # set params
    logging.info(f'Started run.')
    parser = ArgParser()
    zodipol = Zodipol(polarizance=polarizance, fov=parser["fov"], n_polarization_ang=n_polarization_ang, parallel=parser["parallel"], n_freq=parser["n_freq"],
                      planetary=parser["planetary"], isl=parser["isl"], resolution=parser["resolution"], imager_params=parser["imager_params"], solar_cut=10 * u.deg)
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())

    obs = zodipol.create_observation(theta=parser["direction"][0], phi=parser["direction"][1], lonlat=False, new_isl=parser["new_isl"])
    obs = obs.add_direction_uncertainty(parser["fov"], parser["resolution"], parser["direction_uncertainty"])
    obs = obs.add_radial_blur(parser["motion_blur"], parser["resolution"])

    biref_amount = zodipol.imager.get_birefringence_mat(np.pi/2, 'center', flat=True, std=3, inv=True)
    biref_angle = zodipol.imager.get_birefringence_mat(np.pi/4, 'constant', flat=True)
    biref_mueller = zodipol.imager.get_birefringence_mueller_matrix(biref_amount, biref_angle)
    biref_obs = zodipol.imager.apply_birefringence(obs, biref_mueller[:, None, ...])

    # plot unnoised camera response
    camera_intensity = zodipol.make_camera_images(biref_obs, n_realizations=1, add_noise=False)
    generate_video(camera_intensity, f'outputs/satellite_polarizance_{polarizance}_biref_vid.mp4', rate=10)
    pass
