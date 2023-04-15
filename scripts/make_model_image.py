import logging
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from astropy.coordinates import CartesianRepresentation

from zodipol.utils.argparser import ArgParser
from zodipol.zodipol import Zodipol


logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


if __name__ == '__main__':
    # set params
    logging.info(f'Started run.')
    parser = ArgParser()
    nside = 128
    zodipol = Zodipol(polarizance=parser["polarizance"], fov=parser["fov"], n_polarization_ang=parser["n_polarization_ang"], parallel=parser["parallel"], n_freq=parser["n_freq"],
                      planetary=parser["planetary"], isl=parser["isl"], resolution=parser["resolution"], imager_params=parser["imager_params"])
    obs_full = zodipol.create_full_sky_observation(nside=nside, obs_time=parser["obs_time"])
    camera_intensity_full_color = zodipol.make_camera_images_multicolor(obs_full,
                                                                        n_realizations=parser["n_realizations"],
                                                                   add_noise=False)
    skymap = camera_intensity_full_color[..., 0]
    # skymap = zodipol.planetary.make_planets_map(nside, parser["obs_time"], zodipol.wavelength)[:, [1,4,8]]

    nside = hp.npix2nside(skymap.shape[0])
    pixel_arr = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, pixel_arr)
    T, P = np.meshgrid(np.linspace(0, np.pi, 1000), np.linspace(-np.pi, np.pi, 1000))
    interp = griddata((theta, phi), skymap, (np.mod(T, np.pi), np.mod(P, 2 * np.pi)), method='nearest')

    venus = zodipol.planetary._get_planet_location('venus', parser["obs_time"])
    venus_phi, venus_theta = venus.lon.to('rad').value, venus.lat.to('rad').value
    venus_phi = np.mod(venus_phi, np.pi) - np.pi * (venus_phi // np.pi)
    # venus_pix = hp.vec2pix(nside, *list(-venus.xyz.value))
    # venus_theta, venus_phi = hp.pix2ang(nside, venus_pix)

    vmin = np.nanmin(interp)
    vmax = np.nanmax(interp)
    interp_norm = (interp - vmin) / (vmax - vmin)
    interp_norm = np.nan_to_num(interp_norm, nan=0.5)
    interp_norm = np.clip(interp_norm, 0, 1)
    interp_norm = interp_norm ** 0.3

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection='mollweide')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.pcolormesh(-P, np.pi / 2 - T, interp_norm, rasterized=True)
    ax.scatter(-venus_phi, venus_theta, facecolors='none', edgecolors='r', label='Venus')

    ax.arrow(-venus_phi, venus_theta+0.3, 0, 0.1-0.3, head_width=0.05, head_length=0.05, lw=3, fc='r', ec='r')
    ax.text(-venus_phi-0.05, venus_theta+0.3, 'Venus', color='r', fontsize=18, horizontalalignment='right')

    ax.plot([-np.pi + 0.1, np.pi - 0.1], [0, 0], 'c', lw=3)
    ax.text(-np.pi + 0.1, 0.05, 'Ecliptic', color='c', fontsize=18)

    center = np.array((np.pi/2, 0))
    x, y = center[0] + 0.6*np.cos(np.linspace(0, 2*np.pi, 50)), center[1] + 0.8*np.sin(np.linspace(0, 2*np.pi, 50))
    ax.plot(x, y, 'y', lw=3)

    fig.tight_layout()
    plt.savefig('outputs/model_image.pdf', format='pdf', bbox_inches='tight', transparent="True", pad_inches=0.1)
    plt.show()

    pass