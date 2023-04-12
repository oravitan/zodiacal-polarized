import numpy as np
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt

from scipy.interpolate import griddata


def plot_satellite_image(image, title=None, resolution=None, saveto=None, **kwargs):
    title = title if title is not None else ""
    unit = str(image.unit) if hasattr(image, "unit") else ""
    image = image.value if hasattr(image, "value") else image
    if resolution is not None:
        image = image.reshape(resolution)
    plt.imshow(image, cmap="afmhot", **kwargs)
    plt.title(title, fontdict={"fontsize": 16})
    cbar = plt.colorbar()
    cbar.set_label(f"{unit}", fontsize=16)
    plt.axis('off')
    if saveto is not None:
        plt.savefig(saveto, bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()


def plot_satellite_image_indices(image, n_plots, title=None, **kwargs):
    for ii in np.linspace(0, image.shape[-1], n_plots, endpoint=False, dtype=int):
        title_cur = title
        if title is not None:
            title_cur += " at index {}".format(ii)
        plot_satellite_image(image[..., ii], title=title_cur, **kwargs)


def plot_skymap_indices(skymap, n_plots, title=None, **kwargs):
    title = title if title is not None else ""
    for ii in np.linspace(0, skymap.shape[-1], n_plots, endpoint=False, dtype=int):
        plot_skymap(skymap[..., ii], title=title + " at index {}".format(ii), **kwargs)


def plot_skymap(skymap, title=None, saveto=None, **kwargs):
    title = title if title is not None else ""
    hp.mollview(
        skymap,
        title=title,
        unit=str(skymap.unit),
        cmap="afmhot",
        **kwargs
    )
    hp.graticule()
    # matplotlib.rcParams.update({'font.size': 22})
    plt.title(title, fontsize=18)
    plt.rcParams.update({'font.size': 16})
    if saveto is not None:
        plt.savefig(saveto, bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()


def plot_skymap_multicolor(skymap, title=None, saveto=None, colorbar=False, log=False, vmin=None, vmax=None, figsize=(10, 6), **kwargs):
    if log:
        skymap = np.log10(skymap)
        vmin = (np.log10(vmin) if vmin is not None else None)
        vmax = (np.log10(vmax) if vmax is not None else None)

    nside = hp.npix2nside(skymap.shape[0])
    pixel_arr = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, pixel_arr)
    T, P = np.meshgrid(np.linspace(0, np.pi, 1000), np.linspace(-np.pi, np.pi, 1000))
    interp = griddata((theta, phi), skymap, (np.mod(T, np.pi), np.mod(P, 2*np.pi)), method='nearest')

    vmin = (np.nanmin(interp) if vmin is None else vmin)
    vmax = (np.nanmax(interp) if vmax is None else vmax)
    interp_norm = (interp - vmin) / (vmax - vmin)
    interp_norm = np.nan_to_num(interp_norm, nan=0.5)
    interp_norm = np.clip(interp_norm, 0, 1)

    plt.figure(figsize=figsize)
    plt.subplot(projection='mollweide')
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, fontsize=18)
    plt.pcolormesh(-P, np.pi/2 - T, interp_norm, rasterized=True, **kwargs)
    if colorbar:
        cbar = plt.colorbar(orientation='horizontal', pad=0.1, ticks=np.linspace(0, 1, 6))
        cbar.set_label("$MJy/sr$ (log-scale)", fontsize=18)
        cbar.ax.set_xticklabels(np.round(np.linspace(np.nanmin(interp), np.nanmax(interp), 6), 2))
        cbar.ax.xaxis.set_tick_params(labelsize=16)
    plt.tight_layout()
    if saveto is not None:
        plt.savefig(saveto, format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()
