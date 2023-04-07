import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
import optuna
from optuna.visualization import plot_contour, plot_optimization_history

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
    mse_kelsall = np.mean((func - kelsall_125um) ** 2)  # l2 norm
    return mse_kelsall
    # bhat = bhat_distance(theta, func, kelsall_125um)
    # return bhat


def bhat_distance(theta, func1, func2):
    f1 = func1 / np.trapz(func1, theta)
    f2 = func2 / np.trapz(func2, theta)
    bhat_distance = -np.log(np.trapz(np.sqrt(f1 * f2), theta))
    return bhat_distance


def scattering_dop(mueller):
    wanted_dop = -0.33 * np.sin(theta) ** 5
    dop = mueller[..., 0, 0, 1] / mueller[..., 0, 0, 0]
    # return bhat_distance(theta, abs(wanted_dop), abs(dop))
    mse_dop = np.mean((abs(wanted_dop) - abs(dop)) ** 2)  # l2 norm
    return mse_dop

def generate_model(x, spectrum):
    s_min, s_max, big_gamma, small_gamma = x[5], x[6], 1, x[7]
    refractive_index_dict = {x[0] + 1j * x[1]: 1-x[4], x[2] + 1j * x[3]: x[4]}
    psm = ParticleSizeModel(s_min=s_min, s_max=s_max, big_gamma=big_gamma, small_gamma=small_gamma,
                            s_res=200)  # create a particle size model
    mie = MieScatteringModel.train(spectrum, particle_size=psm,
                                   refractive_index_dict=refractive_index_dict)  # train a Mie scattering model
    return mie


def optimization_cost(x):
    # print(f"{datetime.now().strftime('%H:%M:%S.%f')}: Current parameters: {'[' + ', '.join(x.round(5).astype(str)) + ']'}")
    mie = generate_model(x, spectrum)

    # plot the model
    dop = []
    dop_distance = []
    dist_from_kesall = []
    for w in C_w:
        mueller_125um = mie.get_mueller_matrix(w, theta)  # get the scattering
        mie_phase_func_125um = mueller_125um[..., 0, 0, 0]
        cur_dist_from_kesall = distance_from_kelsall(theta, mie_phase_func_125um, c=C_w[w])
        dist_from_kesall.append(cur_dist_from_kesall)
        dop_distance.append(scattering_dop(mueller_125um))
        dop.append(np.max(abs(mueller_125um[..., 0, 0, 1] / mueller_125um[..., 0, 0, 0])))

    regularization_factor = 1
    min_dist = np.mean(dist_from_kesall)
    regularization = np.mean(dop_distance)
    total_cost = min_dist + regularization_factor * regularization
    # print('mean max DoP: ', np.mean(dop), '+-', np.std(dop))  # aiming for a DoP of 0.2
    # print(f"{datetime.now().strftime('%H:%M:%S.%f')}: Cost results: (min_dist={min_dist}) + {regularization_factor}*(regularization={regularization})")
    # print(f"{datetime.now().strftime('%H:%M:%S.%f')}: Cost results: (total_cost={total_cost})")
    # print('--------------------')
    return total_cost


def objective(trial):
    m1_i = trial.suggest_float("m1_i", 1.0, 7.0)
    m1_j = trial.suggest_float("m1_j", 0.0, 2.0)
    m2_i = trial.suggest_float("m2_i", 1.0, 7.0)
    m2_j = trial.suggest_float("m2_j", 0.0, 2.0)
    m2_prc = trial.suggest_float("m2_prc", 0.0, 0.5)
    particle_size_small = trial.suggest_float("particle_size_small", 1e-3, 0.5, log=True)
    particle_size_big = trial.suggest_float("particle_size_big", 0.5, 10, log=True)
    particle_size_power = trial.suggest_float("particle_size_power", 1.0, 5.0)
    x = np.array([m1_i, m1_j, m2_i, m2_j, m2_prc, particle_size_small, particle_size_big, particle_size_power])
    return optimization_cost(x)


if __name__ == '__main__':
    spectrum = np.logspace(np.log10(300), np.log10(1300), 10)  # white light wavelength in nm
    theta = np.linspace(0, np.pi, 100)  # angle in radians

    # optuna.delete_study(study_name='mie_optimization', storage='sqlite:///outputs/mie_optimization.db')
    study = optuna.create_study(study_name='mie_optimization', storage='sqlite:///outputs/mie_optimization.db',
                                sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(),
                                load_if_exists=True)
    # study.enqueue_trial({ "m1_i": 6.11175, "m1_j": 0.08278, "m2_i": 3.07128, "m2_j": 0.00182, "m2_prc": 0.20079, "particle_size_small": 0.10301,
    #         "particle_size_big": 0.75446, "particle_size_power": 1.74194})
    # study.enqueue_trial({'m1_i': 4.27595706533983, 'm1_j': 0.2170492420819936, 'm2_i': 4.95990327842391, 'm2_j': 0.3462876172196013, 'm2_prc': 0.20948996668721834, 'particle_size_small': 0.33357907982230667, 'particle_size_big': 0.7514542362538205, 'particle_size_power': 2.0995733449355694})
    study.optimize(objective, n_trials=1000)
    best_params = study.best_params
    x = [best_params['m1_i'], best_params['m1_j'], best_params['m2_i'], best_params['m2_j'], best_params['m2_prc'], \
        best_params['particle_size_small'], best_params['particle_size_big'], best_params['particle_size_power']]

    plot_optimization_history(study)
    plot_contour(study)

    mie = generate_model(x, spectrum)
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
    mie_wl = generate_model(x, spectrum_wl)
    mie_wl_scatt = mie(spectrum_wl, theta)

    # plot the Mueller matrix elements
    plot_mueller_matrix_elems(theta, mie_wl_scatt[:, :, 0, 0], mie_wl_scatt[:, :, 0, 1], mie_wl_scatt[:, :, 2, 2],
                              mie_wl_scatt[:, :, 2, 3], title='Mueller Matrix Elements in the Visible Spectrum')
    plt.savefig('outputs/mie_mueller_matrix_elems_visible.pdf', format='pdf')

    # save the model
    mie.save(MIE_MODEL_DEFAULT_PATH)
    print(MIE_MODEL_DEFAULT_PATH)
