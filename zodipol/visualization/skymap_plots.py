import numpy as np
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt


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
        plt.savefig(saveto)
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
        plt.savefig(saveto)
    plt.show()

