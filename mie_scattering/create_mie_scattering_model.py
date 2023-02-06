import numpy as np

from mie_scattering.scattering_model import calculate_S1S2
from mie_scattering.solar_irradiance_model import get_wl_spectrum, get_solar_irradiance, get_solar_probability_density, get_dirbe_solar_irradiance
from mie_scattering.particle_size_model import get_particle_size_model
from mie_scattering.plotting import plot_total_intensities, plot_polarizations, plot_intensity_polarization, \
    plot_mueller_matrix_elems
from utils.constants import m_graphite, m_silicate, prc_graphite, prc_silicate


if __name__ == '__main__':
    # get the particle size model
    s, n_norm = get_particle_size_model(s_res=400)  # particle size in nm, normalized probability density

    # get the solar irradiance spectrum model
    spectrum = get_wl_spectrum(n_samples=20).to('nm').value  # wavelength in nm
    solar_irradiance = get_solar_irradiance(spectrum)  # solar irradiance in MJy/sr
    # spectrum, solar_irradiance = get_dirbe_solar_irradiance()  # solar irradiance in MJy/sr
    solar_probability_density = get_solar_probability_density(solar_irradiance, spectrum)  # solar probability density in MJy/sr/nm

    # calculate the scattering functions
    theta = np.linspace(0, np.pi, 361)  # scattering angle in rad
    mu = np.cos(theta)  # cosine of scattering angle
    refractive_ind = {m_graphite: prc_graphite, m_silicate: prc_silicate}
    scat_functions = calculate_S1S2(s, n_norm, spectrum, mu, refractive_ind)

    # plot to total intensities
    plot_total_intensities(scat_functions, theta)

    # plot the polarization
    plot_polarizations(scat_functions, theta)

    # Combine the model weighted based on the solar irradiance
    S1_all = np.sum([sol * scat['S1'] for scat, sol in zip(scat_functions, solar_probability_density)], axis=0)
    S2_all = np.sum([sol * scat['S2'] for scat, sol in zip(scat_functions, solar_probability_density)], axis=0)

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

    # save the results (theta, S11, S12, S33, S34)
    np.savetxt('outputs/mie_scattering.csv', np.vstack((theta, S11, S12, S33, S34)).T, delimiter=',', header='theta,S11,S12,S33,S34')
