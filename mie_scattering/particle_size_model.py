import numpy as np
from utils.math import normalize


s_min = 0.57  # particle min size in um
s_max = 3000  # particle max size in um

astroid_gamma = -4.5
commet_gamma = -2.72
commet_percentage = 0.9


def get_particle_size_model(s_res=300) -> (np.ndarray, np.ndarray):
    """
    Get the particle size model
    :param s_res: particle size resolution
    :return: particle size model
    """
    s = np.logspace(np.log10(s_min), np.log10(s_max), s_res) * 1e3  # particle size in nm

    n_astroid = s ** (astroid_gamma + 1)  # astroidal particle size distribution
    n_comet = s ** (commet_gamma + 1)  # comet particle size distribution
    n = (1 - commet_percentage) * n_astroid + commet_percentage * n_comet  # weighted particle size distribution
    n_norm = normalize(n * np.gradient(s))  # normalized particle size distribution
    return s, n_norm


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
