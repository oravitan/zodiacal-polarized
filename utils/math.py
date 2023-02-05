import numpy as np

def normalize(x):
    """
    Normalize a list
    :param x: list
    :return: normalized list
    """
    return x / np.sum(x)
