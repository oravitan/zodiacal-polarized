import numpy as np
from zodipol.utils.math import normalize


# default parameters
DEFAULT_NUM_PARTICLES = 300  # number of particles in the particle size model
MIN_PARTICLE_SIZE = 0.009  # particle min size in um
MAX_PARTICLE_SIZE = 20  # particle max size in um
BIG_PARTICLES_GAMMA = 5.4  # power law exponent for large particles
SMALL_PARTICLES_GAMMA = 4  # power law exponent for small particles
CUTOFF_POINT = 20  # cutoff point for power law exponent


class ParticleSizeModel:
    """
    Particle size model
    """
    def __init__(self, s_res=DEFAULT_NUM_PARTICLES,
                 s_min=MIN_PARTICLE_SIZE,
                 s_max=MAX_PARTICLE_SIZE,
                 big_gamma=BIG_PARTICLES_GAMMA,
                 small_gamma=SMALL_PARTICLES_GAMMA,
                 cutoff_point=CUTOFF_POINT):
        """
        Initialize the particle size model
        :param s_res: particle size resolution
        """
        self.s_min = s_min  # particle min size in um
        self.s_max = s_max  # particle max size in um
        self.big_gamma = big_gamma  # power law exponent for large particles
        self.small_gamma = small_gamma  # power law exponent for small particles
        self.cutoff_point = cutoff_point  # cutoff point for power law exponent
        self.particle_size, self.particle_likelihood = self._get_particle_size_model(s_res=s_res)  # particle size model (nm)

    def _get_particle_size_model(self, s_res=300) -> (np.ndarray, np.ndarray):
        """
        Get the particle size model
        :param s_res: particle size resolution
        :return: particle size model
        """
        particle_size = np.logspace(np.log10(self.s_min), np.log10(self.s_max), s_res)  # particle size in um
        normalization_factor = self.cutoff_point ** (self.big_gamma - self.small_gamma)
        n = np.piecewise(particle_size, particle_size > self.cutoff_point,
                         (lambda x: normalization_factor * (x ** -self.big_gamma), lambda x: x ** -self.small_gamma))
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
