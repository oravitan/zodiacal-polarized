import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy import Zodipy


if __name__ == '__main__':
    model = Zodipy("dirbe", solar_cut=30 * u.deg, extrapolate=True)  # Initialize the model
    nside = 256  # Healpix resolution
    wavelength = 0.7 * u.um  # Wavelength of the observation
    frequency = wavelength.to(u.THz, equivalencies=u.spectral())  # Frequency of the observation

    print("Frequency: ", frequency, "Wavelength: ", wavelength)

    # Calculate the emission at pixels
    binned_emission = model.get_binned_emission_pix(
        frequency,
        pixels=np.arange(hp.nside2npix(nside)),
        nside=nside,
        obs_time=Time("2022-06-14"),
        obs="earth",
    )

    # Plot the emission
    hp.mollview(
        binned_emission,
        title="Binned zodiacal emission at {}".format(wavelength),
        unit="MJy/sr",
        min=0,
        max=1,
        cmap="afmhot",
    )
    plt.show()
