import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy_local.zodipy import Zodipy


if __name__ == '__main__':
    model = Zodipy("dirbe", solar_cut=30 * u.deg, extrapolate=True, parallel=True)  # Initialize the model
    nside = 256  # Healpix resolution
    # wavelength = 0.7 * u.um  # Wavelength of the observation
    wavelength = 1.25 * u.um  # Wavelength of the observation
    frequency = wavelength.to(u.THz, equivalencies=u.spectral())  # Frequency of the observation

    print("Frequency: ", frequency, "Wavelength: ", wavelength)

    polarizance = 1  # Polarizance of the observation
    polarization_angle = np.linspace(0, np.pi, 4, endpoint=False)  # Polarization angle of the observation

    # Calculate the emission at pixels
    binned_emission = model.get_binned_emission_pix(
        frequency,
        pixels=np.arange(hp.nside2npix(nside)),
        nside=nside,
        obs_time=Time("2022-06-14"),
        obs="earth",
        polarization_angle=polarization_angle,
        polarizance=polarizance,
    )

    # Plot the emission
    for ii in range(binned_emission.shape[1]):
        hp.mollview(
            binned_emission[..., ii],
            title="Binned zodiacal emission at {} with polarization angle {}".format(wavelength, np.round(polarization_angle[ii],2)),
            unit="MJy/sr",
            min=0,
            max=1,
            cmap="afmhot",
            rot=(0, 0, 0)
        )
        hp.graticule()
        plt.show()
