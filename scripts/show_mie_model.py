import numpy as np
import matplotlib.pyplot as plt

from zodipy._source_funcs import get_phase_function
from zodipy.source_params import PHASE_FUNCTION_DIRBE, SPECTRUM_DIRBE
from zodipol.zodipol.zodipol import MIE_MODEL_DEFAULT_PATH

from scripts.generate_mie_optuna import generate_model


wavelength = SPECTRUM_DIRBE[:3].to('nm').value  # in nm
C = list(zip(*PHASE_FUNCTION_DIRBE))[:3]
C_w = dict(zip(wavelength.round().astype(int), C))
C_w = {1250: C_w[1250]}


if __name__ == '__main__':
    res_dict = {'m1_i': 2.7875497292739335, 'm1_j': 1.2596510696681886, 'm1_alpha': 20.134718848398713, 'm1_beta': 0.1884620145158685, 'm1_prc': 0.4252298090439352, 'm2_i': 1.995318559493747, 'm2_j': 0.04338205677980103, 'm2_alpha': 96.9202025970916, 'm2_beta': 0.3721117694697997, 'm2_prc': 0.620480793351804}
    spectrum = np.logspace(np.log10(300), np.log10(1300), 10)  # white light wavelength in nm
    theta = np.linspace(0, np.pi, 100)  # angle in radians

    mie = generate_model(res_dict, spectrum)

    mie_scatt1250 = mie(np.array((1250,)), theta)
    mie_scatt500 = mie(np.array((500,)), theta)
    kelsall_125um = get_phase_function(theta, C_w[1250])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(theta, mie_scatt1250[:, :, 0, 0], lw=3, label='$\Phi_{mie}$')
    ax1.plot(theta, kelsall_125um, 'k', lw=3, label='$\Phi_{kelsall}$')
    plt.suptitle('Comparison of Mie model vs. Kelsall ($1.25\mu m$)\nand Mie DOLP vs. empirical ($0.5\mu m$)',
                  fontsize=18)
    ax1.set_ylabel('Phase Function', fontsize=16)
    ax1.grid()
    ax1.legend(prop={'size': 14})
    ax1.tick_params(axis='both', which='major', labelsize=14)

    ax2.plot(theta, mie_scatt500[:, :, 0, 1] / mie_scatt500[:, :, 0, 0], lw=3, label='$DOLP_{mie}$')
    ax2.plot(theta, -0.33 * np.sin(theta) ** 5, lw=3, label='$DOLP_{emp}$')
    ax2.set_ylabel('DOLP', fontsize=16)
    ax2.legend(prop={'size': 14})
    ax2.grid()
    ax2.set_xlabel('Scattering angle (rad)', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    fig.tight_layout()
    fig.savefig('outputs/mie_optimization_comp.pdf')
    plt.show()

    save_flag = input('Save the model? (y/n)')
    if save_flag == 'y' or save_flag == 'Y' or save_flag == 'yes' or save_flag == 'Yes':
        print('Saving model...')
        wl_spectrum = np.logspace(np.log10(300), np.log10(700), 20)
        mie_wl = generate_model(res_dict, wl_spectrum)
        mie_wl.save(MIE_MODEL_DEFAULT_PATH)
        print('Saved model to: ', MIE_MODEL_DEFAULT_PATH)
    else:
        print('Model saving skipped.')
