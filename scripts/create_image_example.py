import logging

from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol, Observation
from zodipol.visualization.skymap_plots import plot_satellite_image, plot_satellite_image_indices

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


if __name__ == '__main__':
    # set params
    logging.info(f'Started run.')
    parser = ArgParser()

    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"], n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"], n_freq=parser["n_freq"],
                      planetary=parser["planetry"], isl=parser["isl"], resolution=parser["resolution"], imager_params=parser["imager_params"])
    obs = zodipol.create_observation(theta=parser["direction"][0], phi=parser["direction"][1], lonlat=False, new_isl=parser["new_isl"])

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
    camera_intensity = zodipol.make_camera_images(obs, n_realizations=parser["n_realizations"], add_noise=False)
    obs_camera_intensity = Observation.from_image(camera_intensity, parser["polarization_angle"])
    camera_dolp = obs_camera_intensity.get_dolp()
    camera_aop = obs_camera_intensity.get_aop()

    logging.info(f'Plotting the camera intensity of the first polarization angle.')
    plot_satellite_image_indices(camera_intensity, 2, resolution=parser["resolution"], title="Camera Polarized Intensity")

    logging.info(f'Plotting the camera polarization.')
    plot_satellite_image(camera_dolp, resolution=parser["resolution"], title="Camera polarization")
    plot_satellite_image(camera_aop, resolution=parser["resolution"], title="Camera angle of polarization")

    # Calculate the number of photons
    logging.info(f'Calculating the realistic image with noise.')
    camera_intensity_noise = zodipol.make_camera_images(obs, n_realizations=parser["n_realizations"], add_noise=True)
    obs_camera_intensity_noise = Observation.from_image(camera_intensity_noise, parser["polarization_angle"])
    camera_dolp_noise = obs_camera_intensity_noise.get_dolp()
    camera_aop_noise = obs_camera_intensity_noise.get_aop()

    # Plot the emission of the first polarization angle
    logging.info(f'Plotting the camera intensity of the first polarization angle.')
    plot_satellite_image_indices(camera_intensity_noise, 4, resolution=parser["resolution"], title="Camera Noised Polarized Intensity")

    logging.info(f'Plotting the camera polarization.')
    plot_satellite_image(camera_dolp_noise, resolution=parser["resolution"], title="Camera Noised polarization")
    plot_satellite_image(camera_aop_noise, resolution=parser["resolution"], title="Camera Noised angle of polarization")
