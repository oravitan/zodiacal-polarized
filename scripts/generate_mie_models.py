import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime

from zodipy._source_funcs import get_phase_function
from zodipy.source_params import PHASE_FUNCTION_DIRBE, SPECTRUM_DIRBE

from zodipol.mie_scattering.particle_size_model import ParticleSizeModel
from zodipol.mie_scattering.mie_scattering_model import MieScatteringModel
from zodipol.visualization.mie_plotting import plot_mueller_matrix_elems
from zodipol.zodipol.zodipol import MIE_MODEL_DEFAULT_PATH


wavelength = SPECTRUM_DIRBE[:3].to('nm').value  # in nm
C = list(zip(*PHASE_FUNCTION_DIRBE))[:3]
C_w = dict(zip(wavelength.round().astype(int), C))
C_w = {1250: C_w[1250]}


def distance_from_kelsall(theta, func, c):
    kelsall_125um = get_phase_function(theta, c)
    bhat = bhat_distance(theta, func, kelsall_125um)
    return bhat


def bhat_distance(theta, func1, func2):
    bhat_distance = -np.log(2*np.pi*np.trapz(np.sqrt(func1 * func2) * np.sin(theta), theta))
    return bhat_distance


def scattering_dop(mueller):
    return np.max(abs(mueller[..., 0, 1] / mueller[..., 0, 0]))


def optimization_cost(x):
    print(f"{datetime.now().strftime('%H:%M:%S.%f')}: Current parameters: {x}")
    s_min, s_max, big_gamma, small_gamma = x
    psm = ParticleSizeModel(s_min=s_min, s_max=s_max, big_gamma=big_gamma, small_gamma=small_gamma, s_res=200)  # create a particle size model
    mie = MieScatteringModel.train(spectrum, particle_size=psm)  # train a Mie scattering model

    # plot the model
    dop = []
    dist_from_kesall = []
    for w in C_w:
        mueller_125um = mie.get_mueller_matrix(w, theta)  # get the scattering
        mie_phase_func_125um = mueller_125um[..., 0, 0, 0]
        cur_dist_from_kesall = distance_from_kelsall(theta, mie_phase_func_125um, c=C_w[w])
        dist_from_kesall.append(cur_dist_from_kesall)
        dop.append(scattering_dop(mueller_125um))

    regularization_factor = 0.4
    min_dist = np.mean(dist_from_kesall)
    regularization = (np.mean(dop) - 0.2) ** 2  # aiming for a DoP of 0.2
    print('mean DoP:', np.mean(dop))
    print(f"{datetime.now().strftime('%H:%M:%S.%f')}: Cost results: (min_dist={min_dist}) + {regularization_factor}*(regularization={regularization})")
    return min_dist + regularization_factor * regularization


if __name__ == '__main__':
    spectrum = np.logspace(np.log10(300), np.log10(1300), 10)  # white light wavelength in nm
    theta = np.linspace(0, np.pi, 100)  # angle in radians

    # create a parameter mapping
    print('Starting optimization')
    x0 = np.array([1, 10, 4.4, 3])
    x = minimize(optimization_cost, x0, method='Nelder-Mead', tol=0.001, options={'disp': True, 'maxiter': 100, 'adaptive': True})
    print("Finished with x=", x)
    s_min, s_max, big_gamma, small_gamma = x.x
    psm = ParticleSizeModel(s_min=s_min, s_max=s_max, big_gamma=big_gamma, small_gamma=small_gamma)  # create a particle size model
    mie = MieScatteringModel.train(spectrum, particle_size=psm)
    mie_scatt = mie(spectrum, theta)

    # plot the Mueller matrix elements
    plot_mueller_matrix_elems(theta, mie_scatt[:, :, 0, 0], mie_scatt[:, :, 0, 1], mie_scatt[:, :, 2, 2],
                              mie_scatt[:, :, 2, 3], title='Mueller Matrix Elements in wide Spectrum')
    plt.savefig('outputs/mie_mueller_matrix_elems_all.pdf', format='pdf')

    # Compare to Kelsall's model
    plt.figure()
    for n, w in enumerate(C_w):
        mie_scatt = mie(w, theta)
        kelsall = get_phase_function(theta, C_w[w])
        plt.plot(theta, mie_scatt[:, :, 0, 0], color=f'C{n}', lw=3, label=f'Our model ($\lambda={w} nm$)')
        plt.plot(theta, kelsall, '--', color=f'C{n}', lw=3, label=f'Kelsall ($\lambda={w} nm$)')
    plt.title('Comparison of unpolarized phase functions\n in our model vs. Kelsall', fontsize=16)
    plt.xlabel('Scattering angle (rad)', fontsize=14)
    plt.ylabel('Phase function', fontsize=14)
    plt.grid()
    plt.legend()
    plt.savefig('outputs/mie_kelsall_model_compare.pdf', format='pdf')
    plt.show()

    # Display out model in the White Light spectrum
    spectrum_wl = np.logspace(np.log10(300), np.log10(700), 10)  # white light wavelength in nm
    psm = ParticleSizeModel(s_min=s_min, s_max=s_max, big_gamma=big_gamma, small_gamma=small_gamma)  # create a particle size model
    mie_wl = MieScatteringModel.train(spectrum_wl, particle_size=psm)
    mie_wl_scatt = mie(spectrum_wl, theta)

    # plot the Mueller matrix elements
    plot_mueller_matrix_elems(theta, mie_wl_scatt[:, :, 0, 0], mie_wl_scatt[:, :, 0, 1], mie_wl_scatt[:, :, 2, 2],
                              mie_wl_scatt[:, :, 2, 3], title='Mueller Matrix Elements in the Visible Spectrum')
    plt.savefig('outputs/mie_mueller_matrix_elems_visible.pdf', format='pdf')

    # save the model
    mie.save(MIE_MODEL_DEFAULT_PATH)
    print(MIE_MODEL_DEFAULT_PATH)
