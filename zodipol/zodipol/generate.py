import os
import pickle as pkl
import numpy as np
import astropy.units as u
from tqdm import tqdm


DEFAULT_PATH = 'saved_models/self_calibration_40.pkl'


def get_observations(n_rotations, zodipol, parser,  rotations_file_path=DEFAULT_PATH):
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
    n_rotations = len(obs)
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
    polarizance_real = polarizance.reshape((-1, 1, 1)).repeat(parser["n_polarization_ang"], axis=-1)

    obs_orig = [zodipol.make_camera_images(o, polarizance_real, polarization_angle_real, n_realizations=parser["n_realizations"], add_noise=True) for o in obs_biref]
    images_orig = zodipol.post_process_images(np.stack(obs_orig, axis=-1))
    return obs, images_orig, polarizance_real.reshape(-1, parser["n_polarization_ang"]), polarization_angle_spatial_diff.reshape((-1)), mueller_truth

