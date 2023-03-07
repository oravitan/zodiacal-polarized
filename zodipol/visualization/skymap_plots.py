import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def plot_satellite_image(image, title=None, resolution=None, saveto=None, **kwargs):
    title = title if title is not None else ""
    unit = str(image.unit) if hasattr(image, "unit") else ""
    image = image.value if hasattr(image, "value") else image
    if resolution is not None:
        image = image.reshape(resolution)
    plt.imshow(image, cmap="afmhot", **kwargs)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(f"{unit}")
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()


def plot_satellite_image_indices(image, n_plots, title=None, **kwargs):
    title = title if title is not None else ""
    for ii in np.linspace(0, image.shape[-1], n_plots, endpoint=False, dtype=int):
        plot_satellite_image(image[..., ii], title=title + " at index {}".format(ii), **kwargs)


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
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()

