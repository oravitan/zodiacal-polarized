from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt

from zodipol.zodipy_local.zodipy._ipd_model import RRM, InterplanetaryDustModel, Kelsall
from zodipol.zodipy_local.zodipy._source_funcs import (
    get_dust_grain_temperature,
    get_scattering_angle,
)

from zodipol.mie_scattering.mueller_matrices import get_unpolarized_stokes_vector, get_rotation_mueller_matrix
from zodipol.mie_scattering.mie_scattering_model import MieScatteringModel

if TYPE_CHECKING:

    from zodipol.zodipy_local.zodipy._ipd_dens_funcs import ComponentDensityFn

"""
Function that return the zodiacal emission at a step along all lines of sight given 
a zodiacal model.
"""
GetCompEmissionAtStepFn = Callable[..., npt.NDArray[np.float64]]



def kelsall_optical_depth(
    r: npt.NDArray[np.float64],
    start: np.float64,
    stop: npt.NDArray[np.float64],
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    get_density_function: ComponentDensityFn,
    mie_scattering_model: MieScatteringModel,
    wavelength: np.float64,
        **kwargs
) -> npt.NDArray[np.float64]:
    """Kelsall uses common line of sight grid from obs to 5.2 AU."""
    # Convert the quadrature range from [-1, 1] to the true ecliptic positions
    R_los = ((stop - start) / 2) * r + (stop + start) / 2
    X_los = R_los * u_los
    X_helio = X_los + X_obs

    density = get_density_function(X_helio)
    extinction1250 = mie_scattering_model.get_extinction(1.25)[0]
    extinction_w = mie_scattering_model.get_extinction(wavelength)[0]
    return density * extinction_w[None, :] / extinction1250