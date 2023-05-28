from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, TypeVar, Union

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

from zodipol.zodipy_local.zodipy._bandpass import Bandpass
from zodipol.zodipy_local.zodipy._constants import SPECIFIC_INTENSITY_UNITS
from zodipol.zodipy_local.zodipy._ipd_comps import ComponentLabel
from zodipol.zodipy_local.zodipy._ipd_model import RRM, InterplanetaryDustModel, Kelsall
from zodipol.zodipy_local.zodipy._source_funcs import get_phase_function

InterplanetaryDustModelT = TypeVar(
    "InterplanetaryDustModelT", bound=InterplanetaryDustModel
)

"""Return the source parameters for a given bandpass and model. 
Must match arguments in the emission fns."""
GetSourceParametersFn = Callable[
    [Bandpass, InterplanetaryDustModelT], Dict[Union[ComponentLabel, str], Any]
]


def get_source_parameters_kelsall_comp(
    bandpass: Bandpass, model: Kelsall, keep_freq_elems=False
) -> dict[ComponentLabel | str, dict[str, Any]]:
    if not bandpass.frequencies.unit.is_equivalent(model.spectrum.unit):
        bandpass.switch_convention()

    spectrum = (
        model.spectrum.to_value(u.Hz)
        if model.spectrum.unit.is_equivalent(u.Hz)
        else model.spectrum.to_value(u.micron)
    )

    interpolator = partial(interp1d, x=spectrum, fill_value="extrapolate")

    source_parameters: dict[ComponentLabel | str, dict[str, Any]] = {}
    for comp_label in model.comps:
        source_parameters[comp_label] = {}
        emissivity = interpolator(y=model.emissivities[comp_label])(
            bandpass.frequencies.value
        )
        if model.albedos is not None:
            albedo = interpolator(y=model.albedos[comp_label])(
                bandpass.frequencies.value
            )
        else:
            albedo = 0

        if bandpass.frequencies.size > 1 and not keep_freq_elems:
            emissivity = bandpass.integrate(emissivity)
            albedo = bandpass.integrate(albedo)

        source_parameters[comp_label]["emissivity"] = emissivity
        source_parameters[comp_label]["albedo"] = albedo

    phase_interp_ang = np.linspace(0, np.pi, 100)
    phase_coefficients = np.asarray(model.phase_coefficients)
    phase_funcs = interpolate_phase(phase_interp_ang, bandpass.frequencies.value, phase_coefficients, spectrum)

    if model.solar_irradiance is not None:
        solar_irradiance = interpolator(y=model.solar_irradiance)(
            bandpass.frequencies.value
        )
        solar_irradiance = u.Quantity(solar_irradiance, "MJy /sr").to_value(
            SPECIFIC_INTENSITY_UNITS, equivalencies=u.spectral()
        )
    else:
        solar_irradiance = 0

    if bandpass.frequencies.size > 1 and not keep_freq_elems:
        phase_coefficients = bandpass.integrate(phase_coefficients)
        solar_irradiance = bandpass.integrate(solar_irradiance)
    source_parameters["common"] = {}
    source_parameters["common"]["phase_coefficients"] = tuple(phase_funcs)
    source_parameters["common"]["phase_angs"] = phase_interp_ang
    source_parameters["common"]["solar_irradiance"] = solar_irradiance
    source_parameters["common"]["T_0"] = model.T_0
    source_parameters["common"]["delta"] = model.delta

    return source_parameters


def interpolate_phase(interp_angle, wavelength, C, spectrum):
    orig_phase_func = [get_phase_function(interp_angle, c) for c in list(zip(*C))]
    interp_res = []
    for ang, val in zip(interp_angle, list(zip(*orig_phase_func))):
        val_log = np.log10(val)
        interpolator = interp1d(x=spectrum, y=val_log, fill_value="extrapolate")
        ang_res = interpolator(wavelength / 1000)
        interp_res.append(10 ** (ang_res))
    interp_res_wavelength = list(zip(*interp_res))
    return interp_res_wavelength


def get_source_parameters_rmm(
    bandpass: Bandpass, model: RRM
) -> dict[ComponentLabel | str, dict[str, Any]]:
    if not bandpass.frequencies.unit.is_equivalent(model.spectrum.unit):
        bandpass.switch_convention()

    spectrum = (
        model.spectrum.to_value(u.Hz)
        if model.spectrum.unit.is_equivalent(u.Hz)
        else model.spectrum.to_value(u.micron)
    )

    source_parameters: dict[ComponentLabel | str, dict[str, Any]] = {}
    calibration = interp1d(x=spectrum, y=model.calibration, fill_value="extrapolate")(
        bandpass.frequencies.value
    )
    calibration = u.Quantity(calibration, u.MJy / u.AU).to_value(u.Jy / u.cm)

    if bandpass.frequencies.size > 1:
        calibration = bandpass.integrate(calibration)

    for comp_label in model.comps:
        source_parameters[comp_label] = {}
        source_parameters[comp_label]["T_0"] = model.T_0[comp_label]
        source_parameters[comp_label]["delta"] = model.delta[comp_label]

    source_parameters["common"] = {"calibration": calibration}

    return source_parameters


SOURCE_PARAMS_MAPPING: dict[type[InterplanetaryDustModel], GetSourceParametersFn] = {
    Kelsall: get_source_parameters_kelsall_comp,
    RRM: get_source_parameters_rmm,
}
