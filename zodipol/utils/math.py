import numpy as np

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
