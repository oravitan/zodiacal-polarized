import matplotlib.pyplot as plt
import numpy as np


def plot_total_intensities(scat_functions, theta) -> None:
    """
    Plot the scattering functions as a function of the scattering angle
    :param scat_functions: list of scattering functions of different wavelengths
    :param theta: scattering angle
    """
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    for scat in scat_functions:
        SL, SR = np.real(scat['S1'].conj() * scat['S1']), np.real(scat['S2'].conj() * scat['S2'])
        SU = (SL + SR) / 2
        ax1.semilogy(theta, SL, lw=2, label=f"$\lambda={round(scat['w'])} nm$")
        ax2.semilogy(theta, SR, lw=2, label=f"$\lambda={round(scat['w'])} nm$")
        ax3.semilogy(theta, SU, lw=2, label=f"$\lambda={round(scat['w'])} nm$")
    ax1.set_xlabel(r"Scattering Angle (deg)")
    ax1.set_ylabel(r"Parallel Intensity")
    ax2.set_xlabel(r"Scattering Angle (deg)")
    ax2.set_ylabel(r"Perpendicular Intensity")
    ax3.set_xlabel(r"Scattering Angle (deg)")
    ax3.set_ylabel(r"Unpolarized Intensity")
    ax1.grid(True, which="both")
    ax2.grid(True, which="both")
    ax3.grid(True, which="both")
    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    handles, labels = ax3.get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower center', ncol=3)
    fig1.tight_layout(pad=2.0)
    fig1.subplots_adjust(bottom=0.35)
    plt.show()


def plot_polarizations(scat_functions, theta) -> None:
    """
    Plot the polarization as a function of the scattering angle
    :param scat_functions: list of scattering functions of different wavelengths
    :param theta: scattering angle
    """
    fig2 = plt.figure(figsize=(10, 6))
    ax1 = fig2.add_subplot(1, 1, 1)
    for scat in scat_functions:
        SL, SR = np.real(scat['S1'].conj() * scat['S1']), np.real(scat['S2'].conj() * scat['S2'])
        P = (SL - SR) / (SL + SR)
        ax1.plot(theta, P, lw=2, label=f"$\lambda={round(scat['w'])} nm$")
    ax1.set_xlabel(r"Scattering Angle (deg)")
    ax1.set_ylabel(r"Polarization (%)")
    plt.grid()
    handles, labels = ax1.get_legend_handles_labels()
    fig2.legend(handles, labels, loc='center right', ncol=1)
    fig2.subplots_adjust(right=0.85)
    plt.show()


def plot_intensity_polarization(theta, SL, SR, SU, P) -> None:
    """
    Plots the intensity and polarization of a single scattering function on one plot
    :param theta: scattering angle
    :param SL: parallel polarization intensity
    :param SR: perpendicular polarization intensity
    :param SU: unpolarized intensity
    :param P: Polarization
    """
    fig2 = plt.figure(figsize=(10, 6))
    ax1 = fig2.add_subplot(1, 1, 1)
    ax1.semilogy(theta, SL, lw=2, label=f"Parallel Intensity")
    ax1.semilogy(theta, SR, lw=2, label=f"Perpendicular Intensity")
    ax1.semilogy(theta, SU, lw=2, label=f"Unpolarized Intensity")
    plt.grid(True, which="both")
    ax2 = ax1.twinx()
    ax2.plot(theta, P, 'k--', lw=2, alpha=0.5, label='Polarization')
    ax1.set_xlabel(r"Scattering Angle (deg)")
    ax1.set_ylabel(r"Intensity")
    ax2.set_ylabel(r"Polarization (%)")
    fig2.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title('Combined Scattering Function and Polarization Of All Wavelengths and Particle Sizes')
    plt.show()


def plot_mueller_matrix_elems(theta, S11, S12, S33, S34) -> None:
    """
    Plot the Mueller matrix elements as a function of the scattering angle
    :param theta: scattering angle
    :param S11: Mueller matrix element S11=S22
    :param S12: Mueller matrix element S12=S21
    :param S33: Mueller matrix element S33=S44
    :param S34: Mueller matrix element S34=S43
    """
    fig3, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    ax1.plot(theta, S11, lw=2, label='S11')
    ax2.plot(theta, S12, lw=2, label='S12')
    ax3.plot(theta, S33, lw=2, label='S33')
    ax4.plot(theta, S34, lw=2, label='S34')
    ax4.set_xlabel(r"Scattering Angle (deg)")
    ax1.set_ylabel(r"$S_{11}$ Intensity")
    ax2.set_ylabel(r"$S_{12}$ Intensity")
    ax3.set_ylabel(r"$S_{33}$ Intensity")
    ax4.set_ylabel(r"$S_{34}$ Intensity")
    ax1.grid(True, which="both")
    ax2.grid(True, which="both")
    ax3.grid(True, which="both")
    ax4.grid(True, which="both")
    fig3.tight_layout(pad=2.0)
    plt.show()
