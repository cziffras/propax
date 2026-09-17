import jax.numpy as jnp

from .._state import PropertyMap
from ..config import ThermoVar
from ..domain import is_two_phase
from ..interp import BicubicInterpolation
from .results import as_mixed


def in_axis_order(
    table: BicubicInterpolation, tvar1: ThermoVar, val1, tvar2: ThermoVar, val2
):
    """The inputs in the table's own axis order."""
    if tvar1.value == table.x_name:
        return val1, val2
    if tvar2.value == table.x_name:
        return val2, val1
    raise ValueError(
        f"table {table.x_name}-{table.y_name} does not carry the pair "
        f"({tvar1.value}, {tvar2.value})"
    )


def call_interp(interpolators, pair_to_table_map, tvar1, val1, tvar2, val2):
    """
    Interpolated outputs for a pair, on the branch the state belongs to, and
    whether they rest on solved nodes only.
    """
    key = frozenset({tvar1, tvar2})
    if key not in pair_to_table_map:
        raise KeyError(
            f"No interpolation table found for pair {tvar1.value}, {tvar2.value}"
        )

    table_name = pair_to_table_map[key]
    table = interpolators[table_name]
    xy = in_axis_order(table, tvar1, val1, tvar2, val2)
    result_vec = table(*xy)
    extrapolated = table.is_extrapolated(*xy)

    dome = interpolators.get(f"{table_name}_dome")
    if dome is not None:
        # the mixture table carries the quality as its last channel, so the
        # branch is chosen by a lookup; outside the mixture table the
        # interpolant returns NaN and two_phase is false
        dome_xy = in_axis_order(dome, tvar1, val1, tvar2, val2)
        dome_vec = dome(*dome_xy)
        # stored in the form the lever rule is linear in, `as_mixed` turns it back
        dome_vec = jnp.stack(
            [
                as_mixed(ThermoVar(n), dome_vec[i])
                for i, n in enumerate(dome.output_names)
            ]
        )
        dome_extrapolated = dome.is_extrapolated(*dome_xy)
        # an extrapolated quality must never override a solved single-phase cell
        two_phase = is_two_phase(dome_vec[-1], ~dome_extrapolated | extrapolated)
        result_vec = jnp.where(two_phase, dome_vec[:-1], result_vec)
        extrapolated = jnp.where(two_phase, dome_extrapolated, extrapolated)

    # `output_names` holds CoolProp keys, which are the ThermoVar values
    result = {
        ThermoVar(name): result_vec[i] for i, name in enumerate(table.output_names)
    }
    result[tvar1] = jnp.asarray(val1, dtype=result_vec[0].dtype)
    result[tvar2] = jnp.asarray(val2, dtype=result_vec[0].dtype)
    # off the axes the lookup is NaN, which is not reliable either
    reliable = ~extrapolated & jnp.all(jnp.isfinite(result_vec))
    return PropertyMap(result), reliable
