import numpy as np
from zodipol.utils import normalize


s_min = 0.2  # particle min size in um
s_max = 200  # particle max size in um

big_gamma = 4.4  # power law exponent for large particles
small_gamma = 3  # power law exponent for small particles
cutoff_point = 20  # cutoff point for power law exponent


class ParticleSizeModel:
    """
    Particle size model
    """
    def __init__(self, s_res=300):
        """
        Initialize the particle size model
        :param s_res: particle size resolution
        """
        self.particle_size, self.particle_likelihood = self._get_particle_size_model(s_res=s_res)

    @staticmethod
    def _get_particle_size_model(s_res=300) -> (np.ndarray, np.ndarray):
        """
        Get the particle size model
        :param s_res: particle size resolution
        :return: particle size model
        """
        particle_size = np.logspace(np.log10(s_min), np.log10(s_max), s_res)  # particle size in um
        normalization_factor = cutoff_point ** (big_gamma - small_gamma)
        n = np.piecewise(particle_size, particle_size > cutoff_point,
                         (lambda x: normalization_factor * (x ** -big_gamma), lambda x: x ** -small_gamma))
        particle_likelihood = normalize(n * np.gradient(particle_size))  # normalized particle size distribution
        return particle_size * 1e3, particle_likelihood


if __name__ == '__main__':
    import matplotlib.pyplot as plt  # import here to avoid unnecessary import

    psm = ParticleSizeModel()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.semilogy(psm.particle_size, psm.particle_likelihood, 'b', ls='dashdot', lw=1, label="Normalized Particle Size Distribution")
    ax1.set_xlabel(r"Particle Size (um)")
    ax1.set_ylabel(r"Probability Density")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()
