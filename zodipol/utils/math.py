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
