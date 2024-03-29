from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt
from itertools import repeat

from zodipol.zodipy_local.zodipy._ipd_model import RRM, InterplanetaryDustModel, Kelsall
from zodipol.zodipy_local.zodipy._source_funcs import (
    get_dust_grain_temperature,
    get_scattering_angle,
    get_phase_function,
    interpolate_phase
)

from zodipol.utils.mueller_matrices import get_unpolarized_stokes_vector, get_rotation_mueller_matrix

if TYPE_CHECKING:

    from zodipol.zodipy_local.zodipy._ipd_dens_funcs import ComponentDensityFn

"""
Function that return the zodiacal emission at a step along all lines of sight given 
a zodiacal model.
"""
GetCompEmissionAtStepFn = Callable[..., npt.NDArray[np.float64]]



def kelsall(
    r: npt.NDArray[np.float64],
    start: np.float64,
    stop: npt.NDArray[np.float64],
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    get_density_function: ComponentDensityFn,
    T_0: float,
    delta: float,
    emissivity: np.float64 | list[np.float64],
    albedo: np.float64 | list[np.float64],
    phase_coefficients: tuple[float, ...],
    phase_angs: list[np.float64],
    solar_irradiance: np.float64 | list[np.float64],
    bp_interpolation_table: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Kelsall uses common line of sight grid from obs to 5.2 AU."""
    # Convert the quadrature range from [-1, 1] to the true ecliptic positions
    R_los = ((stop - start) / 2) * r + (stop + start) / 2
    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    temperature = get_dust_grain_temperature(R_helio, T_0, delta)
    blackbody_emission = np.interp(temperature, *bp_interpolation_table)
    emission = (emissivity[None, :] * blackbody_emission)

    unpolarized_stokes = get_unpolarized_stokes_vector()
    emission = np.einsum('...j,jkw->...kw', emission[..., None], unpolarized_stokes)

    extinction_sca = get_density_function(X_helio) * albedo[None,:]
    extinction_abs = get_density_function(X_helio) * (1-albedo[None, :])
    emission = emission * extinction_abs[..., None, None]

    if any(albedo != 0):
        solar_flux = solar_irradiance / R_helio**2
        scattering_angle = get_scattering_angle(R_los, R_helio, X_los, X_helio)
        phase_function = interpolate_phase(scattering_angle.squeeze(), phase_coefficients, phase_angs)

        scattering_intensity = np.stack([phase_function, np.zeros_like(phase_function), np.zeros_like(phase_function), np.zeros_like(phase_function)], axis=-1)
        emission += extinction_sca[..., None, None] * solar_flux[..., None, None] * scattering_intensity[..., None]

        scattering_dop = -0.33 * np.sin(scattering_angle.squeeze()) ** 5
        emission[..., 1, :] = emission[..., 0, :] * scattering_dop[..., None, None]  # resulting DOP in the visible domain

    n_sca = X_los / R_los
    n_i = X_helio / R_helio
    A = np.cross(n_sca, n_i, axis=0)
    B = np.cross(A, n_sca, axis=0)
    B /= np.linalg.norm(B, axis=0, keepdims=True)
    x, y, z = B[0, :, 0], B[1, :, 0], B[2, :, 0]
    theta_scat = np.arctan2((x ** 2 + y ** 2) ** 0.5, z)
    camera_rotation_mueller = get_rotation_mueller_matrix(theta_scat)
    emission_rotated = np.einsum('ijk,imkl->imj', camera_rotation_mueller, emission)
    return emission_rotated


def rrm(
    r: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    stop: npt.NDArray[np.float64],
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    get_density_function: ComponentDensityFn,
    T_0: float,
    delta: float,
    calibration: np.float64,
    bp_interpolation_table: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """RRM is implented with component specific line-of-sight grids."""
    # Convert the quadrature range from [-1, 1] to the true ecliptic positions
    R_los = ((stop - start) / 2) * r + (stop + start) / 2
    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    temperature = get_dust_grain_temperature(R_helio, T_0, delta)
    blackbody_emission = np.interp(temperature, *bp_interpolation_table)

    return blackbody_emission * get_density_function(X_helio) * calibration


EMISSION_MAPPING: dict[type[InterplanetaryDustModel], GetCompEmissionAtStepFn] = {
    Kelsall: kelsall,
    RRM: rrm,
}
