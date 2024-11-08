from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import astropy.units as u
import numpy as np

from zodipol.zodipy_local.zodipy._ipd_dens_funcs import construct_density_partials
from zodipol.zodipy_local.zodipy.model_registry import model_registry

if TYPE_CHECKING:
    import numpy.typing as npt

DEFAULT_EARTH_POS = u.Quantity([1, 0, 0], u.AU)


def tabulate_density(
    grid: npt.NDArray[np.floating] | Sequence[npt.NDArray[np.floating]],
    model: str = "DIRBE",
    earth_pos: u.Quantity[u.AU] = DEFAULT_EARTH_POS,
) -> npt.NDArray[np.float64]:
    """Return the tabulated densities of the zodiacal components for a given grid.

    Args:
        grid: A cartesian mesh grid (x, y, z).
        model: Name of interplanetary dust model supported by ZodiPy.
        earth_pos: Position of the Earth in AU.

    Returns:
        density_grid: The tabulated zodiacal component densities.

    """
    ipd_model = model_registry.get_model(model)

    if not isinstance(grid, np.ndarray):
        grid = np.asarray(grid)

    # Prepare attributes and variables for broadcasting with the grid
    earth_position = np.reshape(earth_pos.to(u.AU).value, (3, 1, 1, 1))
    for comp in ipd_model.comps.values():
        comp.X_0 = np.reshape(comp.X_0, (3, 1, 1, 1))

    partials = construct_density_partials(
        list(ipd_model.comps.values()), {"X_earth": earth_position}
    )

    density_grid = np.zeros((len(ipd_model.comps), *grid.shape[1:]))
    for idx, partial in enumerate(partials):
        density_grid[idx] = partial(grid)

    # Revert broadcasting reshapes
    for comp in ipd_model.comps.values():
        comp.X_0 = np.reshape(comp.X_0, (3, 1))

    return density_grid
