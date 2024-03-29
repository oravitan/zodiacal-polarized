import logging
import astropy.units as u
import numpy as np

from zodipol.utils.argparser import ArgParser
from zodipol.zodipol.zodipol import Zodipol
from zodipol.zodipol.observation import Observation
from zodipol.visualization.skymap_plots import plot_satellite_image, plot_satellite_image_indices, plot_skymap

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


if __name__ == '__main__':
    # set params
    logging.info(f'Started run.')
    parser = ArgParser()
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"], n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"], n_freq=parser["n_freq"],
                      planetary=parser["planetary"], isl=parser["isl"], resolution=parser["resolution"], imager_params=parser["imager_params"], solar_cut=10 * u.deg)
    A_gamma = zodipol.imager.get_A_gamma(zodipol.frequency, zodipol.get_imager_response())
    # obs_full = zodipol.create_full_sky_observation(nside=128, obs_time=parser["obs_time"])
    # camera_intensity_full_color = zodipol.make_camera_images_multicolor(obs_full,
    #                                                                     n_realizations=parser["n_realizations"],
    #                                                                     add_noise=False)
    # obs_camera_intensity_full_color = Observation.from_image(camera_intensity_full_color, parser["polarizance"],
    #                                                          parser["polarization_angle"][None, None, :])
    # camera_dolp_color = obs_camera_intensity_full_color.get_dolp()
    # plot_skymap(camera_intensity_full_color[..., 0, 0] / A_gamma, format='%.2f')
    # plot_skymap(camera_dolp_color[..., 0], format='%.2f', saveto=f'outputs/camera_dolp.pdf')

    obs = zodipol.create_observation(theta=parser["direction"][0], phi=parser["direction"][1], lonlat=False, new_isl=True)
    obs = obs.add_direction_uncertainty(parser["fov"], parser["resolution"], parser["direction_uncertainty"])
    obs = obs.add_radial_blur(parser["motion_blur"], parser["resolution"])

    # Calculate the polarization
    logging.info(f'Calculating the polarization.')
    binned_emission = obs.get_binned_emission(parser["polarization_angle"], parser["polarizance"])
    binned_dolp = obs.get_dolp()
    binned_aop = obs.get_aop()

    # Plot the emission of the first polarization angle
    logging.info(f'Plotting the emission of the first polarization angle.')
    plot_satellite_image_indices(binned_emission[..., -1, :], 2, resolution=parser["resolution"], title="Binned zodiacal emission")

    # plot the binned polarization
    logging.info(f'Plotting the binned polarization.')
    plot_satellite_image(binned_dolp[..., -1], resolution=parser["resolution"], title="Binned zodiacal polarization")
    plot_satellite_image(binned_aop[..., -1], resolution=parser["resolution"], title="Binned angle of polarization")

    # plot unnoised camera response
    camera_intensity = zodipol.make_camera_images(obs, n_realizations=1, add_noise=False)
    obs_camera_intensity = Observation.from_image(camera_intensity, parser["polarizance"], parser["polarization_angle"][None, :])
    camera_dolp = obs_camera_intensity.get_dolp()
    camera_aop = obs_camera_intensity.get_aop()

    print('Median number of electrons: ', np.median(camera_intensity / A_gamma))

    logging.info(f'Plotting the camera intensity of the first polarization angle.')
    plot_satellite_image_indices(camera_intensity / A_gamma, 2, resolution=parser["resolution"], title="Camera Polarized Intensity")

    logging.info(f'Plotting the camera polarization.')
    plot_satellite_image(camera_dolp, resolution=parser["resolution"], title="Camera polarization")
    plot_satellite_image(camera_aop, resolution=parser["resolution"], title="Camera angle of polarization")

    # Calculate the number of photons
    logging.info(f'Calculating the realistic image with noise.')
    camera_intensity_noise = zodipol.make_camera_images(obs, n_realizations=parser["n_realizations"], add_noise=True)
    obs_camera_intensity_noise = Observation.from_image(camera_intensity_noise, parser["polarizance"], parser["polarization_angle"][None, :])
    camera_dolp_noise = obs_camera_intensity_noise.get_dolp()
    camera_aop_noise = obs_camera_intensity_noise.get_aop()

    print('Median number of electrons: ', np.median(camera_intensity_noise / A_gamma))

    # Plot the emission of the first polarization angle
    logging.info(f'Plotting the camera intensity of the first polarization angle.')
    plot_satellite_image_indices(camera_intensity_noise / A_gamma, 2, resolution=parser["resolution"], title="Camera Noised Polarized Intensity")

    logging.info(f'Plotting the camera polarization.')
    plot_satellite_image(camera_dolp_noise, resolution=parser["resolution"], title="Camera Noised polarization")
    plot_satellite_image(camera_aop_noise, resolution=parser["resolution"], title="Camera Noised angle of polarization")

    # Add imager birefringence to the received image
    logging.info(f'Adding imager birefringence to the received image.')
    biref_amount = zodipol.imager.get_birefringence_mat(0.1, 'center', flat=True, std=3)
    biref_angle = zodipol.imager.get_birefringence_mat(0.1, 'constant', flat=True)
    biref_mueller = zodipol.imager.get_birefringence_mueller_matrix(biref_amount, biref_angle)
    biref_obs = zodipol.imager.apply_birefringence(obs, biref_mueller[:, None, ...])

    logging.info(f'Plotting the biref camera intensity.')
    biref_intensity_noise = zodipol.make_camera_images(biref_obs, n_realizations=parser["n_realizations"], add_noise=True)
    plot_satellite_image_indices(biref_intensity_noise, 4, resolution=parser["resolution"],title="Camera Noised Polarized Intensity")

    # make color images
    import matplotlib.pyplot as plt
    gamma = 0.6
    # camera_intensity_color = zodipol.make_camera_images_multicolor(obs, n_realizations=1, add_noise=False)
    camera_intensity_color = binned_emission[:, [5, 12, 22], :]
    I_color = camera_intensity_color.reshape((300, 200, 3, 4))
    I_color_norm = (I_color - I_color.min()) / (I_color.max() - I_color.min())
    I_color_norm = I_color_norm ** gamma

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(I_color_norm[..., 0]); ax[0, 0].axis('off')
    ax[0, 1].imshow(I_color_norm[..., 1]); ax[0, 1].axis('off')
    ax[1, 0].imshow(I_color_norm[..., 3]); ax[1, 0].axis('off')
    ax[1, 1].imshow(I_color_norm[..., 2]); ax[1, 1].axis('off')
    fig.tight_layout()
    plt.savefig('outputs/image_exmaple_color.pdf', format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()
    pass
