from typing import Mapping

import jax.numpy as jnp
from jaxtyping import Array

from .._state import PropertyMap, SaturationResult, TxResult
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


def properties_Tx(saturation: Superancillary, T: Array, x: Array) -> TxResult:
    sat = saturation.state_T(T)

    v_L = 1.0 / sat.L[ThermoVar.D]
    v_V = 1.0 / sat.V[ThermoVar.D]
    rho_mix = 1.0 / ((1.0 - x) * v_L + x * v_V)

    mix = {
        ThermoVar.D: rho_mix,
        ThermoVar.U: (1.0 - x) * sat.L[ThermoVar.U] + x * sat.V[ThermoVar.U],
        ThermoVar.H: (1.0 - x) * sat.L[ThermoVar.H] + x * sat.V[ThermoVar.H],
        ThermoVar.S: (1.0 - x) * sat.L[ThermoVar.S] + x * sat.V[ThermoVar.S],
    }
    return TxResult(mix=PropertyMap(mix), L=sat.L, V=sat.V, x=x, T=T, P=sat.P)


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


# Heat capacities have no value in a two-phase state, CoolProp uses the lever rule
# to fail silently, propax return NaNs
UNDEFINED_TWO_PHASE = (ThermoVar.CVMASS, ThermoVar.CPMASS)


def undefine_two_phase(result: PropertyMap, dtype) -> PropertyMap:
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    return result.replace({k: nan for k in UNDEFINED_TWO_PHASE})


def fill_result_dict(partial_result: Mapping, dtype, transport: bool = False) -> dict:
    """
    Missing entries are padded with a finite 0.0 rather than NaN: an unused NaN
    sitting in the returned pytree might poison reverse-mode autodiff (0 * NaN = NaN
    in the cotangent).

    That is for absent entries (those which could not be solved); one that exists and has no
    value says so with NaN (see `undefine_two_phase`).
    """
    pad_val = jnp.asarray(0.0, dtype=dtype)
    keys = RESULT_KEYS["common"] + RESULT_KEYS["single_phase"]
    keys = keys + RESULT_KEYS["transport"] if transport else keys
    return {key: partial_result.get(key.internal_key, pad_val) for key in keys}


def get_sat_bounds(sat_state: SaturationResult, tvar: ThermoVar):
    if not tvar.spec.can_be_input:
        raise ValueError(f"Variable {tvar.value} cannot be an input")

    if tvar.spec.sat_key_unique:
        # `tvar.value` is the CoolProp key; SaturationResult carries the same
        # names, see _state
        val = getattr(sat_state, tvar.value)
        return (1.0 / val, 1.0 / val) if tvar.spec.invert_for_mixing else (val, val)

    if tvar.spec.sat_key_L and tvar.spec.sat_key_V:
        L_val, V_val = sat_state.L[tvar], sat_state.V[tvar]
        if tvar.spec.invert_for_mixing:
            return 1.0 / L_val, 1.0 / V_val
        return L_val, V_val

    raise ValueError(f"Saturation metadata missing for {tvar.value}")
