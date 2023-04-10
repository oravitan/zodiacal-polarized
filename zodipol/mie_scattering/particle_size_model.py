import numpy as np
from scipy.stats import gamma

from zodipol.utils.math import normalize


# default parameters
MIN_PARTICLE_SIZE = 50  # particle min size in nm
MAX_PARTICLE_SIZE = 50000  # particle max size in nm
PARTICLE_SIZE_RES = 200  # particle size resolution


class ParticleModel:
    """
    Particle size model
    """
    def __init__(self, refractive_index, alpha, beta, s_res=PARTICLE_SIZE_RES):
        """
        Initialize the particle size model
        :param s_res: particle size resolution
        """
        self.refractive_index = refractive_index
        self.alpha = alpha
        self.beta = beta
        self.particle_size, self.particle_likelihood = self._get_particle_size_model(s_res=s_res)  # particle size model (nm)

    def _get_particle_size_model(self, s_res=300) -> (np.ndarray, np.ndarray):
        """
        Get the particle size model
        :param s_res: particle size resolution
        :return: particle size model
        """
        particle_size = np.logspace(np.log10(MIN_PARTICLE_SIZE), np.log10(MAX_PARTICLE_SIZE), s_res)  # particle size in um
        gamma_dist = gamma(a=self.alpha, scale=1/self.beta).pdf(particle_size)  # gamma distribution
        return particle_size, gamma_dist

    def get_particle_size_prc(self, size):
        return np.interp(size, self.particle_size, self.particle_likelihood)


class ParticleTable:
    def __init__(self, particle_model: list, particle_percentage: list):
        """
        Initialize the particle size table
        :param particle_size: particle size in um
        :param particle_likelihood: particle likelihood
        """
        self._validate_inputs(particle_model, particle_percentage)
        self.particle_model = particle_model
        self.particle_percentage = particle_percentage

    def get_model_list(self):
        return zip(self.particle_model, self.particle_percentage)

    @staticmethod
    def _validate_inputs(particle_model, particle_percentage):
        assert len(particle_model) == len(particle_percentage), "The length of particle model and particle percentage should be the same"
        assert abs(np.sum(particle_percentage)-1) < 1e-5, "The sum of particle percentage should be 1"
        assert all([isinstance(particle, ParticleModel) for particle in particle_model]), "The particle model should be a list of ParticleModel"

    def get_particle_size_prc(self, size):
        particle_prc_list = []
        for particle, prc in zip(self.particle_model, self.particle_percentage):
            particle_prc = particle.get_particle_size_prc(size)
            particle_prc_list.append(particle_prc * prc)
        return particle_prc_list


particle_graphite = ParticleModel(refractive_index=1.8+0.1j, alpha=1.5, beta=0.5)
particle_silicate = ParticleModel(refractive_index=1.8+0.1j, alpha=1.5, beta=0.5)
DEFAULT_PARTICLE_MODEL = ParticleTable(particle_model=[particle_graphite, particle_silicate],particle_percentage=[0.2, 0.8])


if __name__ == '__main__':
    import matplotlib.pyplot as plt  # import here to avoid unnecessary import

    psm = ParticleModel()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.semilogy(psm.particle_size, psm.particle_likelihood, 'b', ls='dashdot', lw=1, label="Normalized Particle Size Distribution")
    ax1.set_xlabel(r"Particle Size (um)")
    ax1.set_ylabel(r"Probability Density")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()
