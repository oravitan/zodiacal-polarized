import numpy as np

from mie_scattering.mie_scattering_model import MieScatteringModel
from mie_scattering.solar_irradiance_model import SolarIrradianceModel
from mie_scattering.particle_size_model import ParticleSizeModel
from mie_scattering.plotting import plot_total_intensities, plot_polarizations, plot_intensity_polarization, \
    plot_mueller_matrix_elems
from utils.constants import refractive_ind


if __name__ == '__main__':
    # get the particle size model
    psm = ParticleSizeModel()  # particle size model
    sim = SolarIrradianceModel()  # solar irradiance model

    mie_scatt_wavelength = {}
    for w in sim.spectrum:
        mie_scatt_wavelength[w] = MieScatteringModel(refractive_ind, psm, w)

    # plot to total intensities
    plot_total_intensities(mie_scatt_wavelength)

    # plot the polarization
    plot_polarizations(mie_scatt_wavelength)

    # Combine the model weighted based on the solar irradiance
    init_model = list(mie_scatt_wavelength.values())[0]
    theta = init_model.theta
    S1_all = np.sum([sol * scat.S1 for scat, sol in zip(mie_scatt_wavelength.values(), sim.solar_likelihood)], axis=0)
    S2_all = np.sum([sol * scat.S2 for scat, sol in zip(mie_scatt_wavelength.values(), sim.solar_likelihood)], axis=0)

    SL, SR = np.real(S1_all.conj() * S1_all), np.real(S2_all.conj() * S2_all)
    SU = (SL + SR) / 2
    P = (SL - SR) / (SL + SR)

    # plot the combined results
    plot_intensity_polarization(theta, SL, SR, SU, P)

    # calculate Mie scattering Mueller matrix elements
    S11 = 0.5 * (np.real(S2_all.conj() * S2_all) + np.real(S1_all.conj() * S1_all))
    S12 = 0.5 * (np.real(S2_all.conj() * S2_all) - np.real(S1_all.conj() * S1_all))
    S33 = np.real(0.5 * (S2_all.conj() * S1_all + S1_all.conj() * S2_all))
    S34 = np.real(1j * 0.5 * (S2_all.conj() * S1_all - S1_all.conj() * S2_all))

    # Plot the Mueller matrix elements
    plot_mueller_matrix_elems(theta, S11, S12, S33, S34)


