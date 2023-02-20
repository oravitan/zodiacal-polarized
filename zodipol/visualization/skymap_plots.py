import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def plot_skymap_indices(skymap, n_plots, title=None, **kwargs):
    title = title if title is not None else ""
    for ii in np.linspace(0, skymap.shape[-1], n_plots, endpoint=False, dtype=int):
        plot_skymap(skymap[..., ii], title=title + " at index {}".format(ii), **kwargs)


def plot_skymap(skymap, title=None, **kwargs):
    title = title if title is not None else ""
    hp.mollview(
        skymap,
        title=title,
        unit=str(skymap.unit),
        cmap="afmhot",
        **kwargs
    )
    hp.graticule()
    plt.show()

