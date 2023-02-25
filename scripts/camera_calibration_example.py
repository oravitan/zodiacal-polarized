import astropy.units as u
import numpy as np
import logging
from astropy.time import Time
from scipy.optimize import least_squares

from zodipol.zodipol import Zodipol, Observation

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


if __name__ == '__main__':
    # set params
    logging.info(f'Started run.')
    nside = 64  # Healpix resolution
    polarizance = 1  # Polarizance of the observation
    n_polarization_angles = 4
    fov = 5  # deg

    polarization_shift = 0.1
    polarization_angle = polarization_shift + np.linspace(0, np.pi, n_polarization_angles, endpoint=False)  # Polarization angle of the observation
    polarization_angle_diff = np.diff(polarization_angle)[0]

    # Initialize the model
    logging.info(f'Initializing the model.')
    zodipol = Zodipol(polarizance=1., fov=5 * u.deg, n_polarization_ang=n_polarization_angles, parallel=True,
                      n_freq=20, planetary=True, resolution=(50, 40))
    obs = zodipol.create_observation(theta=np.pi / 2 * u.deg, phi=0 * u.deg, lonlat=False, new_isl=False)

    logging.info(f'Create real images.')
    df_ind = polarization_angle_diff * np.arange(n_polarization_angles)
    make_img = lambda eta: zodipol.make_camera_images(obs, polarizance, eta+df_ind, add_noise=False)

    polarization_angle_res = []
    for ii in range(100):
        camera_intensity_real = zodipol.make_camera_images(obs, polarizance, polarization_angle, add_noise=True)
        cost_function = lambda eta: 1e23 * (make_img(eta) - camera_intensity_real).value.flatten()
        res = least_squares(cost_function, x0=0, bounds=(-np.pi/2, np.pi/2))
        polarization_angle_res.append(res.x)
    pass
