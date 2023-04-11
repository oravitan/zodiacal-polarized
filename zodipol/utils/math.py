import numpy as np
import transformations as transf

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
