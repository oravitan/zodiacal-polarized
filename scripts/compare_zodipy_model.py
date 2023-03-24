import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy.zodipy import Zodipy
from zodipy import Zodipy as Zodipy_unpol


if __name__ == '__main__':
    # set params
    nside = 256  # Healpix resolution
    central_wavelength = 1.3 * u.um  # Wavelength of the observation
    polarizance = 0  # Polarizance of the observation
    polarization_angle = 0  # Polarization angle of the observation

    # Initialize the model
    model_pol = Zodipy("dirbe", solar_cut=30 * u.deg, extrapolate=True, parallel=True)  # Initialize the model
    model_unpol = Zodipy_unpol("dirbe", solar_cut=30 * u.deg, extrapolate=True, parallel=True)  # Initialize the model

    frequency = central_wavelength.to(u.THz, equivalencies=u.spectral())  # Frequency of the observation

    # Calculate the emission at pixels
    binned_emission_pol = model_pol.get_binned_emission_pix(
        frequency,
        pixels=np.arange(hp.nside2npix(nside)),
        nside=nside,
        obs_time=Time("2022-06-14"),
        obs="earth",
        polarization_angle=polarization_angle,
        polarizance=polarizance)
    binned_emission_unpol = model_unpol.get_binned_emission_pix(
        frequency,
        pixels=np.arange(hp.nside2npix(nside)),
        nside=nside,
        obs_time=Time("2022-06-14"),
        obs="earth")

    # Plot the emission of the unpolarized model
    hp.mollview(
        binned_emission_unpol,
        title="Binned zodiacal emission at {} of the unpolarized original model".format(central_wavelength),
        unit=str(binned_emission_unpol.unit),
        norm='log',
        cmap="afmhot",
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()

    # Plot the emission of the polarized model
    hp.mollview(
        binned_emission_pol.squeeze(),
        title="Binned zodiacal emission at {} of the polarized model with P=0".format(central_wavelength),
        unit=str(binned_emission_pol.unit),
        norm='log',
        cmap="afmhot",
        rot=(0, 0, 0)
    )
    hp.graticule()
    plt.show()



