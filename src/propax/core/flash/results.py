from typing import Mapping

import jax.numpy as jnp
from jaxtyping import Array

from .._state import PropertyMap, SaturationResult
from ..config import ThermoVar
from ..saturation import Superancillary

RESULT_KEYS = {  # keys output by flash
    "common": [
        ThermoVar.P,
        ThermoVar.T,
        ThermoVar.D,
        ThermoVar.U,
        ThermoVar.H,
        ThermoVar.S,
    ],
    "single_phase": [ThermoVar.CVMASS, ThermoVar.CPMASS],
    "transport": [ThermoVar.VISCOSITY, ThermoVar.CONDUCTIVITY],
}


def mixture_state(saturation: Superancillary, T: Array, x: Array) -> PropertyMap:
    sat = saturation.state_T(T)
    v = (1.0 - x) / sat.L[ThermoVar.D] + x / sat.V[ThermoVar.D]
    undefined = jnp.full_like(v, jnp.nan)
    return PropertyMap(
        {
            ThermoVar.D: 1.0 / v,
            **{
                var: (1.0 - x) * sat.L[var] + x * sat.V[var]
                for var in (ThermoVar.U, ThermoVar.H, ThermoVar.S)
            },
            ThermoVar.P: sat.P,
            ThermoVar.T: T,
            ThermoVar.CVMASS: undefined,
            ThermoVar.CPMASS: undefined,
        }
    )


def add_transport(viscosity, conductivity, result: PropertyMap, transport: bool):
    if not transport:
        return result
    rho, T = result[ThermoVar.D], result[ThermoVar.T]
    return result.replace(
        {
            ThermoVar.VISCOSITY: viscosity.viscosity_rhoT(rho, T),
            ThermoVar.CONDUCTIVITY: conductivity.conductivity_rhoT(rho, T),
        }
    )


def fill_result_dict(partial_result: Mapping, dtype, transport: bool = False) -> dict:
    """
    Missing entries are padded with a finite 0.0 rather than NaN: an unused NaN
    sitting in the returned pytree might poison reverse-mode autodiff (0 * NaN = NaN
    in the cotangent).

    That is for absent entries (those which could not be solved); one that exists and has no
    value says so with NaN (see `mixture_state`).
    """
    pad_val = jnp.asarray(0.0, dtype=dtype)
    keys = RESULT_KEYS["common"] + RESULT_KEYS["single_phase"]
    keys = keys + RESULT_KEYS["transport"] if transport else keys
    return {key: partial_result.get(key.internal_key, pad_val) for key in keys}


def as_mixed(tvar: ThermoVar, value):
    """The form the lever rule is linear in: specific volume for a density."""
    return 1.0 / value if tvar.spec.invert_for_mixing else value


def get_sat_bounds(sat_state: SaturationResult, tvar: ThermoVar):
    if not tvar.spec.can_be_input:
        raise ValueError(f"Variable {tvar.value} cannot be an input")

    if tvar.spec.sat_key_unique:
        # `tvar.value` is the CoolProp key; SaturationResult carries the same
        # names, see _state
        value = as_mixed(tvar, getattr(sat_state, tvar.value))
        return value, value

    if tvar.spec.sat_key_L and tvar.spec.sat_key_V:
        return as_mixed(tvar, sat_state.L[tvar]), as_mixed(tvar, sat_state.V[tvar])

    raise ValueError(f"Saturation metadata missing for {tvar.value}")
