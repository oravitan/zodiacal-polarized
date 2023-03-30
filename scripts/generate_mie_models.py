import numpy as np
import pandas as pd
import os
from tqdm import  tqdm
import matplotlib.pyplot as plt

from zodipy._source_funcs import get_phase_function
from zodipy.source_params import PHASE_FUNCTION_DIRBE, SPECTRUM_DIRBE

from zodipol.mie_scattering.particle_size_model import ParticleSizeModel
from zodipol.mie_scattering.mie_scattering_model import MieScatteringModel
from zodipol.visualization.mie_plotting import plot_mueller_matrix_elems
from zodipol.zodipol.zodipol import MIE_MODEL_DEFAULT_PATH


wavelength = SPECTRUM_DIRBE[:3].to('nm').value  # in nm
C = list(zip(*PHASE_FUNCTION_DIRBE))[:3]
C_w = dict(zip(wavelength.round().astype(int), C))


def distance_from_kelsall(theta, func, c):
    kelsall_125um = get_phase_function(theta, c)
    bhat = bhat_distance(theta, func, kelsall_125um)
    return bhat


def bhat_distance(theta, func1, func2):
    bhat_distance = -np.log(2*np.pi*np.trapz(np.sqrt(func1 * func2) * np.sin(theta), theta))
    return bhat_distance


if __name__ == '__main__':
    spectrum = np.logspace(np.log10(300), np.log10(3500), 20)  # white light wavelength in nm
    theta = np.linspace(0, np.pi, 100)  # angle in radians

    # create a parameter mapping
    s_min = np.logspace(np.log10(0.0001), np.log10(0.001), 10)  # in um
    s_max = np.logspace(np.log10(40), np.log10(60), 5)  # in um
    big_gamma = np.linspace(2, 4, 5)
    small_gamma = np.linspace(3, 5, 12)
    BG, SG, SN, SX = np.meshgrid(big_gamma, small_gamma, s_min, s_max)  # create a grid of parameters
    SN, SX, BG, SG = SN.flatten(), SX.flatten(), BG.flatten(), SG.flatten()  # flatten the grid

    parameter_mapping = pd.DataFrame({'s_min': SN, 's_max': SX, 'big_gamma': BG, 'small_gamma': SG})  # create a dataframe from the grid

    results = []
    for ind, row in tqdm(parameter_mapping.iterrows(), total=len(parameter_mapping)):
        # create a Mie scattering model
        psm = ParticleSizeModel(s_min=row['s_min'], s_max=row['s_max'], big_gamma=row['big_gamma'],
                                small_gamma=row['small_gamma'])  # create a particle size model
        mie = MieScatteringModel.train(spectrum, particle_size=psm)  # train a Mie scattering model

        # plot the model
        dist_from_kesall = {}
        for w in C_w:
            mueller_125um = mie.get_mueller_matrix(w, theta)  # get the scattering
            mie_phase_func_125um = mueller_125um[..., 0, 0, 0]
            cur_dist_from_kesall = distance_from_kelsall(theta, mie_phase_func_125um, c=C_w[w])
            dist_from_kesall[w] = cur_dist_from_kesall

        mueller_spec = mie.get_mueller_matrix(np.array((spectrum.min(), spectrum.max())), theta)  # get the scattering
        mie_phase_func_spec = mueller_spec[..., 0, 0]
        dist_between_spec = bhat_distance(theta, mie_phase_func_spec[..., 0], mie_phase_func_spec[..., -1])

        row_dict = row.to_dict()
        for w in dist_from_kesall:
            row_dict[f'dist_from_kesall_{w}'] = dist_from_kesall[w]
        row_dict['dist_between_spec'] = dist_between_spec
        results.append(row_dict)
    results_df = pd.DataFrame(results)
    results_df = results_df.assign(dist_from_kesall=results_df[['dist_from_kesall_1250', 'dist_from_kesall_2200', 'dist_from_kesall_3500']].mean(axis=1))
    results_df.to_csv('outputs/mie_parameter_results.csv', index=False)

    results_df.plot.scatter(x='dist_from_kesall', y='dist_between_spec', grid=True, loglog=True)
    plt.savefig('outputs/mie_dist_from_kesall.pdf', format='pdf')
    plt.show()

    best_res = results_df.loc[results_df.dist_from_kesall.argmin()].to_dict()
    print('Best result: ', best_res)

    psm = ParticleSizeModel(s_min=best_res['s_min'], s_max=best_res['s_max'], big_gamma=best_res['big_gamma'], small_gamma=best_res['small_gamma'])  # create a particle size model
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
    psm = ParticleSizeModel(s_min=best_res['s_min'], s_max=best_res['s_max'], big_gamma=best_res['big_gamma'],
                            small_gamma=best_res['small_gamma'])  # create a particle size model
    mie_wl = MieScatteringModel.train(spectrum_wl, particle_size=psm)
    mie_wl_scatt = mie(spectrum_wl, theta)

    # plot the Mueller matrix elements
    plot_mueller_matrix_elems(theta, mie_wl_scatt[:, :, 0, 0], mie_wl_scatt[:, :, 0, 1], mie_wl_scatt[:, :, 2, 2],
                              mie_wl_scatt[:, :, 2, 3], title='Mueller Matrix Elements in the Visible Spectrum')
    plt.savefig('outputs/mie_mueller_matrix_elems_visible.pdf', format='pdf')

    # save the model
    mie.save(MIE_MODEL_DEFAULT_PATH)
    print(MIE_MODEL_DEFAULT_PATH)
