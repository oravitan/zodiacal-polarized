import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.visualization import plot_contour, plot_optimization_history

from zodipy._source_funcs import get_phase_function
from zodipy.source_params import PHASE_FUNCTION_DIRBE, SPECTRUM_DIRBE

from zodipol.mie_scattering.particle_size_model import ParticleModel, ParticleTable
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


def bhat_distance(theta, func1, func2):
    f1 = func1 / np.trapz(func1, theta)
    f2 = func2 / np.trapz(func2, theta)
    bhat_distance = -np.log(np.trapz(np.sqrt(f1 * f2), theta))
    return bhat_distance


def scattering_dop(mueller):
    wanted_dop = -0.33 * np.sin(theta) ** 5
    dop = mueller[..., 0, 0, 1] / mueller[..., 0, 0, 0]
    # return bhat_distance(theta, abs(wanted_dop), abs(dop))
    mse_dop = np.mean((wanted_dop - dop) ** 2)  # l2 norm
    return mse_dop


def generate_model(optimization_input, spectrum):
    particle1 = ParticleModel(optimization_input["m1_i"] + 1j * optimization_input["m1_j"],
                              optimization_input["m1_alpha"], optimization_input["m1_beta"])
    particle2 = ParticleModel(optimization_input["m2_i"] + 1j * optimization_input["m2_j"],
                              optimization_input["m2_alpha"], optimization_input["m2_beta"])
    particle_table = ParticleTable([particle1, particle2], [optimization_input["m1_prc"], 1-optimization_input["m1_prc"]])

    mie = MieScatteringModel.train(spectrum, particle_table=particle_table)  # train a Mie scattering model
    return mie


def optimization_cost(optimization_input):
    mie = generate_model(optimization_input, spectrum)

    # plot the model
    dop = []
    dop_distance = []
    dist_from_kesall = []
    for w in C_w:
        mueller_125um = mie.get_mueller_matrix(np.array((w,)), theta)  # get the scattering
        mueller_05um = mie.get_mueller_matrix(np.array((500, )), theta)

        cur_dist_from_kesall = distance_from_kelsall(theta, mueller_125um[..., 0, 0, 0], c=C_w[w])
        dist_from_kesall.append(cur_dist_from_kesall)
        dop_distance.append(scattering_dop(mueller_05um))
        dop.append(np.max(abs(mueller_05um[..., 0, 0, 1] / mueller_05um[..., 0, 0, 0])))

    regularization_factor = 0.5
    min_dist = np.mean(dist_from_kesall)
    regularization = np.mean(dop_distance)
    total_cost = min_dist + regularization_factor * regularization
    return total_cost


def objective(trial):
    optimization_input = dict(
        m1_i=trial.suggest_float("m1_i", 2.0, 3.0),  # graphite
        m1_j=trial.suggest_float("m1_j", 1.0, 1.5),
        m1_alpha=trial.suggest_float("m1_alpha", 0.005, 200, log=True),
        m1_beta=trial.suggest_float("m1_beta", 0.01, 10.0, log=True),
        m1_prc=trial.suggest_float("m1_prc", 0.0, 0.5),

        m2_i=trial.suggest_float("m2_i", 1.3, 2.0), # silicate
        m2_j=trial.suggest_float("m2_j", 0.0, 0.5),
        m2_alpha=trial.suggest_float("m2_alpha", 0.005, 200, log=True),
        m2_beta=trial.suggest_float("m2_beta", 0.01, 10.0, log=True)
    )
    return optimization_cost(optimization_input)


if __name__ == '__main__':
    spectrum = np.logspace(np.log10(300), np.log10(1300), 10)  # white light wavelength in nm
    theta = np.linspace(0, np.pi, 100)  # angle in radians

    # optuna.delete_study(study_name='mie_optimization', storage='sqlite:///db.sqlite3')
    study = optuna.create_study(study_name='mie_optimization', storage='sqlite:///db.sqlite3',
                                sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(),
                                load_if_exists=True)
    study.optimize(objective, n_trials=500)
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
