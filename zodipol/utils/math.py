import numpy as np
import transformations as transf
import astropy.units as u

from functools import lru_cache
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist

xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]


def normalize(x):
    """
    Normalize a list
    :param x: list
    :return: normalized list
    """
    return x / np.sum(x)


def ang2vec(theta, phi):
    """
    Convert spherical coordinates to cartesian coordinates.
    :param theta: polar angle
    :param phi: azimuthal angle
    :return: cartesian coordinates in the form of a numpy array of shape (..., 3)
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def vec2ang(arr):
    """
    Convert cartesian coordinates to spherical coordinates.
    :param arr: cartesian coordinates in the form of a numpy array of shape (..., 3)
    :return: polar angle and azimuthal angle
    """
    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    return theta, phi


def get_c2w(theta, phi, roll):
    """
    Get the transformation matrix from the camera frame to the world frame.
    :param theta: polar angle
    :param phi: azimuthal angle
    :param roll: roll angle
    :return: transformation matrix from the camera frame to the world frame
    """
    rot_c_vec = ang2vec(theta, phi)
    if theta == np.pi/2 and phi == 0.0:
        rot_mat = np.identity(4)
    else:
        rot_mat = transf.rotation_matrix(transf.angle_between_vectors(xaxis, rot_c_vec),
                                         transf.vector_product(xaxis, rot_c_vec))
    roll_mat = transf.rotation_matrix(roll, rot_c_vec)
    trans_mat = transf.concatenate_matrices(roll_mat, rot_mat)
    return trans_mat[:3, :3]


def get_w2c(theta, phi, roll):
    """
    Get the transformation matrix from the world frame to the camera frame.
    :param theta: polar angle
    :param phi: azimuthal angle
    :param roll: roll angle
    :return: transformation matrix from the world frame to the camera frame
    """
    return np.linalg.inv(get_c2w(theta, phi, roll))


# image alignment and rotation methods
def align_images(zodipol, parser, images_res, rotation_arr, invert=False, fill_value=0):
    """
    Align the images to the reference image.
    :param images_res: The images to align.
    :param rotation_arr: The rotation angles of the images relative to reference frame.
    """
    if invert:
        rotation_arr = rotation_arr
    res_images = []
    for ii in range(len(rotation_arr)):
        rot_image = get_rotated_image(zodipol, parser, images_res[..., ii], 0, rotation_arr[ii], fill_value=fill_value)
        res_images.append(rot_image)
    images_res = np.stack(res_images, axis=-1)
    return images_res


def get_rotated_image(zodipol, parser, images, rotation_from, rotation_to, fill_value=0, how='nearest'):
    """
    Rotate the images to the reference frame.
    :param images: The images to rotate.
    :param rotation_to: The rotation angle to rotate to.
    :param fill_value: The value to fill non-intersecting pixels.
    :param how: The interpolation method.
    """
    images = np.nan_to_num(images, nan=fill_value)  # fill nans
    if rotation_to == rotation_from:  # avoid interpolation issues
        return images

    if how == 'linear':
        return _rotate_linear(images, parser, rotation_from,  rotation_to, zodipol, fill_value=fill_value)
    elif how == 'nearest':
        return _rotate_nearest(images, parser, rotation_from, rotation_to, zodipol, fill_value=fill_value)
    else:
        raise ValueError("Invalid interpolation method")


def _rotate_nearest(images, parser, rotation_from, rotation_to, zodipol, fill_value=0):
    """
    Rotate the images to the reference frame using nearest neighbor interpolation.
    :param images: The images to rotate.
    :param rotation_to: The rotation angle to rotate to.
    :param zodipol: The zodipol object.
    :param fill_value: The value to fill non-intersecting pixels.
    """
    x_ind, y_ind = _get_rotation_coords(parser, zodipol, rotation_from, rotation_to)
    # x_ind, y_ind, index_mask = _get_rotation_coords(parser, zodipol, rotation_from, rotation_to)

    images_resh = images.reshape(parser["resolution"] + list(images.shape[1:]))
    # images_interp = np.full_like(images, fill_value)
    images_interp = images_resh[y_ind, x_ind, ...]
    # images_interp[index_mask, ...] = images_resh[y_ind[index_mask], x_ind[index_mask], ...]
    return images_interp


def _rotate_linear(images, parser, rotation_from, rotation_to, zodipol, fill_value=0):
    """
    Rotate the images to the reference frame using linear interpolation.
    :param images: The images to rotate.
    :param rotation_to: The rotation angle to rotate to.
    :param zodipol: The zodipol object.
    :param fill_value: The value to fill non-intersecting pixels.
    """
    theta_from, phi_from = _get_cached_coords(*parser["direction"], rotation_from * u.deg, tuple(parser["resolution"]), zodipol)
    theta_to, phi_to = _get_cached_coords(*parser["direction"], rotation_to * u.deg, tuple(parser["resolution"]), zodipol)

    vec_from = ang2vec(theta_from, phi_from)
    x = np.linspace(vec_from[:, 2].max(), vec_from[:, 2].min(), parser["resolution"][1])
    y = np.linspace(vec_from[:, 1].min(), vec_from[:, 1].max(), parser["resolution"][0])
    vec_to = ang2vec(theta_to, phi_to)
    images_interp_res = images.reshape(parser["resolution"] + list(images.shape[1:]))
    grid_interp = RegularGridInterpolator((y, x), images_interp_res, bounds_error=False, fill_value=fill_value,
                                          method='linear')
    interp = grid_interp(list(zip(vec_to[:, 1], vec_to[:, 2])))
    return interp


@lru_cache
def _get_rotation_coords(parser, zodipol, rotation_from, rotation_to):
    """
    Get the coordinates of the rotated image.
    :param parser: The parser object.
    :param zodipol: The zodipol object.
    :param rotation_to: The rotation angle to rotate to.
    """
    theta_from, phi_from = _get_cached_coords(*parser["direction"], rotation_from * u.deg, tuple(parser["resolution"]), zodipol)
    theta_to, phi_to = _get_cached_coords(*parser["direction"], rotation_to * u.deg, tuple(parser["resolution"]), zodipol)

    vec_from = ang2vec(theta_from, phi_from)
    vec_to = ang2vec(theta_to, phi_to)

    d = cdist(vec_from, vec_to, metric='cosine')
    nearest_ind = np.argmin(d, axis=1)
    X, Y = np.meshgrid(np.arange(0, 200, 1), np.arange(0, 300, 1))
    x_ind = X.flatten()[nearest_ind]
    y_ind = Y.flatten()[nearest_ind]

    # dx = (vec_from[:, 2].max() - vec_from[:, 2].min()) / parser["resolution"][1]
    # dy = (vec_from[:, 0].max() - vec_from[:, 0].min()) / parser["resolution"][0]
    # x_ind = np.round((vec_from[:, 2].max() - vec_to[:, 2]) / dx).astype(int)
    # y_ind = np.round((vec_to[:, 1] - vec_from[:, 1].min()) / dy).astype(int)
    # index_mask = (x_ind >= 0) & (x_ind < parser["resolution"][1]) & (y_ind >= 0) & (y_ind < parser["resolution"][0])
    # return x_ind, y_ind, index_mask
    return x_ind, y_ind


@lru_cache
def _get_cached_coords(theta, phi, roll, resolution, zodipol):
    """
    Get the coordinates of the image (cached function to improve calculation time).
    :param theta: The theta angle.
    :param phi: The phi angle.
    :param roll: The roll angle.
    :param resolution: The resolution of the image.
    :param zodipol: The zodipol object.
    """
    return zodipol.create_sky_coords(theta=theta, phi=phi, roll=roll, resolution=resolution)

