import numpy as np
from utils.math import normalize


s_min = 0.2  # particle min size in um
s_max = 200  # particle max size in um

big_gamma = 4.4  # power law exponent for large particles
small_gamma = 3  # power law exponent for small particles
cutoff_point = 20  # cutoff point for power law exponent


def get_particle_size_model(s_res=300) -> (np.ndarray, np.ndarray):
    """
    Get the particle size model
    :param s_res: particle size resolution
    :return: particle size model
    """
    s = np.logspace(np.log10(s_min), np.log10(s_max), s_res)  # particle size in um
    normalization_factor = cutoff_point ** (big_gamma - small_gamma)
    n = np.piecewise(s, s > cutoff_point, (lambda x: normalization_factor * (x ** -big_gamma), lambda x: x ** -small_gamma))
    n_norm = normalize(n * np.gradient(s))  # normalized particle size distribution
    return s * 1e3, n_norm


if __name__ == '__main__':
    import matplotlib.pyplot as plt  # import here to avoid unnecessary import

    s, n_norm = get_particle_size_model()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.semilogy(s, n_norm, 'b', ls='dashdot', lw=1, label="Normalized Particle Size Distribution")
    ax1.set_xlabel(r"Particle Size (um)")
    ax1.set_ylabel(r"Probability Density")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()
