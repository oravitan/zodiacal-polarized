import numpy as np
import transformations as transf
import astropy.units as u

from scipy.interpolate import RegularGridInterpolator

xaxis, yaxis, zaxis = [1, 0, 0], [0, 1, 0], [0, 0, 1]


def normalize(x):
    """
    Normalize a list
    :param x: list
    :return: normalized list
    """
    return x / np.sum(x)


def make_sorted(ll, *args):
    ll_sorted = np.argsort(ll)
    ll = ll[ll_sorted]
    args = [arg[ll_sorted] for arg in args]
    return ll, *args


# vector calculations
def get_rotation_matrix(v, theta):
    x, y, z = v
    W = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * W @ W


def ang2vec(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def vec2ang(arr):
    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    return theta, phi


def get_c2w(theta, phi, roll):
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
    return np.linalg.inv(get_c2w(theta, phi, roll))


# image alignment and rotation methods
def align_images(zodipol, parser, images_res, rotation_arr, invert=False, fill_value=0, method="nearest"):
    """
    Align the images to the reference image.
    :param images_res: The images to align.
    :param rotation_arr: The rotation angles of the images relative to reference frame.
    """
    if invert:
        rotation_arr = rotation_arr
    res_images = []
    for ii in range(len(rotation_arr)):
        rot_image = get_rotated_image(zodipol, parser, images_res[..., ii], -rotation_arr[ii], fill_value=fill_value, method=method)
        res_images.append(rot_image)
    images_res = np.stack(res_images, axis=-1)
    return images_res


def get_rotated_image(zodipol, parser, images, rotation_to, fill_value=0, method="nearest"):
    """
    Rotate the images to the reference frame.
    :param images: The images to rotate.
    :param rotation_to: The rotation angle to rotate to.
    :param fill_value: The value to fill non-intersecting pixels.
    :param method: The interpolation method.
    """
    images = np.nan_to_num(images, nan=fill_value)  # fill nans
    if rotation_to == 0:  # avoid interpolation issues
        return images
    theta_from, phi_from = zodipol.create_sky_coords(theta=parser["direction"][0],
                                                          phi=parser["direction"][1],
                                                          roll=0 * u.deg, resolution=parser["resolution"])
    vec_from = ang2vec(theta_from, phi_from)
    x = np.linspace(vec_from[:, 0].min(), vec_from[:, 0].max(), parser["resolution"][1])
    y = np.linspace(vec_from[:, 1].min(), vec_from[:, 1].max(), parser["resolution"][0])
    images_resh = images.reshape(parser["resolution"] + list(images.shape[1:]))
    grid_interp = RegularGridInterpolator((y, x), images_resh, bounds_error=False, fill_value=fill_value, method=method)

    theta_to, phi_to = zodipol.create_sky_coords(theta=parser["direction"][0], phi=parser["direction"][1],
                                                      roll=rotation_to * u.deg, resolution=parser["resolution"])
    vec_to = ang2vec(theta_to, phi_to)
    interp = grid_interp(list(zip(vec_to[:, 1], vec_to[:, 0])))
    return interp
