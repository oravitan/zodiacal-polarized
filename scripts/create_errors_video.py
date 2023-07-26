import logging
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io

from zodipol.utils.argparser import ArgParser
from zodipol.zodipol.zodipol import Zodipol


logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def create_images(n_polarization_ang, polarizance_list, dolp_level=0.8):
    parser = ArgParser()
    zodipol = Zodipol(polarizance=1, fov=parser["fov"], n_polarization_ang=n_polarization_ang,
                      parallel=parser["parallel"], n_freq=parser["n_freq"],
                      planetary=parser["planetary"], isl=parser["isl"], resolution=parser["resolution"],
                      imager_params=parser["imager_params"], solar_cut=10 * u.deg)

    obs = zodipol.create_observation(theta=parser["direction"][0], phi=parser["direction"][1], lonlat=False,
                                     new_isl=True)
    obs = obs.add_direction_uncertainty(parser["fov"], parser["resolution"], parser["direction_uncertainty"])
    obs = obs.add_radial_blur(parser["motion_blur"], parser["resolution"])
    dolp_factor = dolp_level / obs.get_dolp().max().value
    obs.Q, obs.U = dolp_factor * obs.Q, dolp_factor * obs.U
    camera_res = []
    for polarizance in polarizance_list:
        camera_intensity = zodipol.make_camera_images(obs, n_realizations=1, add_noise=False, polarizance=polarizance)
        camera_res.append(camera_intensity)
    return np.stack(camera_res, axis=-1)


def create_biref_images(n_polarization_ang, polarizance, dolp_level=0.8):
    parser = ArgParser()
    zodipol = Zodipol(polarizance=1, fov=parser["fov"], n_polarization_ang=n_polarization_ang,
                      parallel=parser["parallel"], n_freq=parser["n_freq"],
                      planetary=parser["planetary"], isl=parser["isl"], resolution=parser["resolution"],
                      imager_params=parser["imager_params"], solar_cut=10 * u.deg)

    obs = zodipol.create_observation(theta=parser["direction"][0], phi=parser["direction"][1], lonlat=False,
                                     new_isl=True)
    obs = obs.add_direction_uncertainty(parser["fov"], parser["resolution"], parser["direction_uncertainty"])
    obs = obs.add_radial_blur(parser["motion_blur"], parser["resolution"])
    dolp_factor = dolp_level / obs.get_dolp().max().value / 2
    obs.Q, obs.U = dolp_factor * obs.Q, dolp_factor * obs.U
    obs.I /= 2

    camera_img = zodipol.make_camera_images(obs, n_realizations=1, add_noise=False, polarizance=polarizance)
    biref_images = []
    for angle in np.linspace(0, np.pi / 2, n_polarization_ang, endpoint=False):
        biref_amount = zodipol.imager.get_birefringence_mat(np.pi / 2, 'constant', flat=True)
        biref_angle = zodipol.imager.get_birefringence_mat(angle, 'constant', flat=True)
        biref_mueller = zodipol.imager.get_birefringence_mueller_matrix(biref_amount, biref_angle)
        biref_obs = zodipol.imager.apply_birefringence(obs, biref_mueller[:, None, ...])
        biref_intensity = zodipol.make_camera_images(biref_obs, n_realizations=1, add_noise=False, polarizance=polarizance)
        biref_images.append(biref_intensity)
    return camera_img, np.stack(biref_images, axis=-1)


def make_comp_vids(arr_list, name_list, resolution=None):
    arr_log = [np.log10(arr) for arr in arr_list]
    arr_min, arr_max = np.min(arr_log), np.max(arr_log)
    arr_norm = [(arr - arr_min) / (arr_max - arr_min) for arr in arr_log]

    for arr, name in zip(arr_norm, name_list):
        generate_video(arr, name, rate=10, stretch=False, resolution=resolution)


def generate_video(camera_res_arr, saveto, rate=10, stretch=True, resolution=None):
    resolution = resolution or parser["resolution"]
    if stretch:
        camera_res_chs = np.log10(camera_res_arr.value)
        camera_res_norm = (camera_res_chs - camera_res_chs.min()) / (camera_res_chs.max() - camera_res_chs.min())
    else:
        camera_res_norm = camera_res_arr.copy()

    frames = []  # for storing the generated images
    for ii in range(camera_res_norm.shape[-1]):
        fig = plt.figure()
        plt.imshow(camera_res_norm[..., ii].reshape(resolution), vmin=0, vmax=1, cmap='gray')
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
    parser = ArgParser()
    n_polarization_ang = 30
    polarization_angles = np.linspace(0, np.pi, n_polarization_ang, endpoint=False)
    camera_intensity = create_images(n_polarization_ang, [1, 0.6])
    no_biref_image, biref_images = create_biref_images(n_polarization_ang, 1, dolp_level=0.8)

    camera_intensity_resh = camera_intensity[..., 0].reshape(parser["resolution"] + [n_polarization_ang, ])
    camera_intensity_resh_deg = camera_intensity[..., -1].reshape(parser["resolution"] + [n_polarization_ang, ])

    indy, indx = 150, 100
    plt.figure()
    plt.plot(polarization_angles, camera_intensity_resh[indy, indx, :], lw=3, label='Polarizance = 1')
    plt.plot(polarization_angles, camera_intensity_resh_deg[indy, indx, :], lw=3, label='Polarizance = 0.6')
    plt.grid()
    # plt.legend(fontsize=16)
    plt.xlabel('Polarization angle', fontsize=18); plt.ylabel('Intensity', fontsize=18)
    plt.gca().tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(f'outputs/satellite_polarizance_comp_{indy}_{indx}.svg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    make_comp_vids([camera_intensity[...,0].value, camera_intensity[..., -1].value], [f'outputs/satellite_polarizance_{1}_vid.mp4', f'outputs/satellite_polarizance_{0.6}_vid.mp4'])

    biref_intensity_resh = biref_images.reshape(parser["resolution"] + [n_polarization_ang, n_polarization_ang, ]).value
    no_biref_intensity_resh = no_biref_image[..., 0, None].repeat(30, axis=-1).reshape(parser["resolution"] + [n_polarization_ang, ]).value

    make_comp_vids([biref_intensity_resh[..., 0, :], no_biref_intensity_resh], ['outputs/satellite_biref_vid.mp4', 'outputs/satellite_no_biref_vid.mp4'])
