from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt

from zodipy_local._ipd_model import RRM, InterplanetaryDustModel, Kelsall
from zodipy_local._source_funcs import (
    get_dust_grain_temperature,
    get_scattering_angle,
)

from zodipol.mie_scattering.mueller_matrices import get_unpolarized_stokes_vector, get_rotation_mueller_matrix
from zodipol.mie_scattering.mie_scattering_model import MieScatteringModel

if TYPE_CHECKING:

    from zodipy_local._ipd_dens_funcs import ComponentDensityFn

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
    emissivity: np.float64,
    albedo: np.float64,
    phase_coefficients: tuple[float, ...],
    solar_irradiance: np.float64,
    bp_interpolation_table: npt.NDArray[np.float64],
    mie_scattering_model: MieScatteringModel,
) -> npt.NDArray[np.float64]:
    """Kelsall uses common line of sight grid from obs to 5.2 AU."""
    # Convert the quadrature range from [-1, 1] to the true ecliptic positions
    R_los = ((stop - start) / 2) * r + (stop + start) / 2
    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    temperature = get_dust_grain_temperature(R_helio, T_0, delta)
    blackbody_emission = np.interp(temperature, *bp_interpolation_table)
    emission = (1 - albedo) * (emissivity * blackbody_emission)

    unpolarized_stokes = get_unpolarized_stokes_vector()
    emission = np.einsum('ij,jkw->ikw', emission, unpolarized_stokes)

    if albedo != 0:
        solar_flux = solar_irradiance / R_helio**2
        scattering_angle = get_scattering_angle(R_los, R_helio, X_los, X_helio)

        scattering_emission = mie_scattering_model.get_mueller_matrix(scattering_angle.squeeze())
        scattering_intensity = np.einsum('ijk,kw->ijw', scattering_emission, unpolarized_stokes[0, ...])
        emission += albedo * solar_flux[..., None] * scattering_intensity
    emission_density = emission * get_density_function(X_helio)[..., None]

    n_sca = X_los / R_los
    n_i = X_helio / R_helio
    A = np.cross(n_sca, n_i, axis=0)
    B = np.cross(A, n_sca, axis=0)
    B /= np.linalg.norm(B, axis=0, keepdims=True)
    x, y, z = B[0, :, 0], B[1, :, 0], B[2, :, 0]
    theta_scat = np.arctan2((x ** 2 + y ** 2) ** 0.5, z)
    camera_rotation_mueller = get_rotation_mueller_matrix(theta_scat)
    emission_rotated = np.einsum('ijk,ikl->ij', camera_rotation_mueller, emission_density)
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
