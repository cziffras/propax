import logging

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from ...core.config import ThermoVar, get_chunk_size  # noqa: E402
from ...core.saturation import Superancillary  # noqa: E402
from ...fluids._registry import EQS_REGISTRY  # noqa: E402

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _make_outputs_fn(eos, viscosity, conductivity, outputs):
    """scalar function (rho, T) -> stacked vector of all output channels"""

    def outputs_fn(rho, T):
        props = eos.props_rhoT(rho, T)
        vals = []
        for var in outputs:
            if var == ThermoVar.VISCOSITY:
                vals.append(viscosity.viscosity_rhoT(rho, T))
            elif var == ThermoVar.CONDUCTIVITY:
                vals.append(conductivity.conductivity_rhoT(rho, T))
            else:
                vals.append(props[var.internal_key])
        return jnp.stack(vals)

    return outputs_fn


def _get_fluid_modules(fluid_name: str):
    processed = fluid_name.lower().replace(" ", "")
    if processed not in EQS_REGISTRY:
        available = ", ".join(sorted(EQS_REGISTRY.keys()))
        raise ValueError(
            f"Fluid '{fluid_name}' not found in registry. Available fluids: {available}"
        )
    eos_factory, _, visc_factory, cond_factory = EQS_REGISTRY[processed]
    eos, viscosity = eos_factory(), visc_factory()
    conductivity = cond_factory(eos=eos, viscosity=viscosity)
    return (
        processed,
        eos,
        viscosity,
        conductivity,
        Superancillary.create(eos, processed),
    )


def _mapped(fn, *arrays, vfn=None, chunk: int | None = None):
    """
    Applies alternatively `vmap` or `lax.map` to input fn (scalar function).
    """
    xs = tuple(jnp.asarray(a) for a in arrays)
    override = get_chunk_size()
    chunk = override if override is not None else chunk
    if chunk is None:
        vfn = jax.jit(jax.vmap(fn)) if vfn is None else vfn
        out = vfn(*xs)
    else:
        out = jax.jit(lambda ts: jax.lax.map(lambda t: fn(*t), ts, batch_size=chunk))(
            xs
        )
    return jax.tree_util.tree_map(np.asarray, out)
