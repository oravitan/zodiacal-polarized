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
    res_dict = {'m1_i': 2.8119006025243616, 'm1_j': 1.1474049990237336, 'm1_alpha': 94.72107084788712, 'm1_beta': 5.371143616847252, 'm1_prc': 0.8653740351012064, 'm2_i': 1.9730985115528372, 'm2_j': 0.08253845687928743, 'm2_alpha': 231.27541016418317, 'm2_beta': 2.689445903431446, 'm2_prc': 0.570359692197511}
    spectrum = np.logspace(np.log10(300), np.log10(3500), 20)  # white light wavelength in nm
    theta = np.linspace(0, np.pi, 100)  # angle in radians

    mie = generate_model(res_dict, spectrum)

    mie_scatt1250 = mie(1250, theta)
    mie_scatt500 = mie(500, theta)
    kelsall_125um = get_phase_function(theta, C_w[1250])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(theta, mie_scatt1250[:, :, 0, 0], lw=3, label='$\Phi_{mie}$')
    ax1.plot(theta, kelsall_125um, 'k', lw=3, label='$\Phi_{kelsall}$')
    ax1.set_title('Comparison of Mie unpolarized phase function vs. Kelsall ($1.25\mu m$)\nand Mie DOLP vs. empirical ($0.5\mu m$)',
                  fontsize=16)
    ax1.set_ylabel('Phase Function', fontsize=14)
    ax1.grid()
    ax1.legend(prop={'size': 14})
    ax1.tick_params(axis='both', which='major', labelsize=14)

    ax2.plot(theta, -mie_scatt500[:, :, 0, 1] / mie_scatt500[:, :, 0, 0], lw=3, label='$DOLP_{mie}$')
    ax2.plot(theta, 0.33 * np.sin(theta) ** 5, lw=3, label='$DOLP_{emp}$')
    ax2.set_ylabel('DOLP', fontsize=16)
    ax2.legend(prop={'size': 14})
    ax2.grid()
    ax2.set_xlabel('Scattering angle (rad)', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    fig.tight_layout()
    plt.savefig('outputs/mie_optimization_comp.pdf')
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
