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


def _make_input_fn(eos, tvar: ThermoVar):
    """scalar function (rho, T) -> value of the input variable"""
    if tvar == ThermoVar.D:
        return lambda rho, T: rho
    if tvar == ThermoVar.T:
        return lambda rho, T: T
    key = tvar.internal_key
    return lambda rho, T: eos.props_rhoT(rho, T)[key]


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


def fill_ghost(arr_x1, arr_x2, f, dx1, dx2, dx1dx2, ok):
    """Grow the solved region by one node in every direction, by a Taylor step.

    A cell straddling the edge of a branch has corners where that branch has no
    solution, so its stencil is incomplete (for the bicubic that requires four
    nodes) and the interpolant returns NaN even for a query point sitting well
    inside the branch. Ghost values are inputed NOT to reflect physical accuracy
    but to close this gap.

    Fields are (n1, n2, channels), the mask `ok` is (n1, n2), channels is typically
    2.
    """
    f, dx1, dx2, dx1dx2, ok = (a.copy() for a in (f, dx1, dx2, dx1dx2, ok))
    h1 = np.diff(arr_x1)[:, None, None]
    h2 = np.diff(arr_x2)[None, :, None]
    lo, hi, all_ = slice(0, -1), slice(1, None), slice(None)

    # each entry of is one direction : tgt, src, sx, sy
    # tgt is modified along x1 and
    steps = [
        ((hi, all_), (lo, all_), h1, 0.0),
        ((lo, all_), (hi, all_), -h1, 0.0),
        ((all_, hi), (all_, lo), 0.0, h2),
        ((all_, lo), (all_, hi), 0.0, -h2),
        ((hi, hi), (lo, lo), h1, h2),
        ((lo, lo), (hi, hi), -h1, -h2),
        ((hi, lo), (lo, hi), h1, -h2),
        ((lo, hi), (hi, lo), -h1, h2),
    ]

    grown = ok.copy()
    out = [a.copy() for a in (f, dx1, dx2, dx1dx2)]
    for tgt, src, sx, sy in steps:
        take = (  # (n1-1, n2)
            ~ok[tgt] & ok[src] & ~grown[tgt]
        )  # a node is modified only if non valid or not already a grown one
        if not take.any():
            continue

        m = take[..., None]  # (n1-1, n2, 1)
        out[0][tgt] = np.where(m, f[src] + dx1[src] * sx + dx2[src] * sy, out[0][tgt])
        out[1][tgt] = np.where(m, dx1[src] + dx1dx2[src] * sy, out[1][tgt])
        out[2][tgt] = np.where(m, dx2[src] + dx1dx2[src] * sx, out[2][tgt])
        out[3][tgt] = np.where(m, dx1dx2[src], out[3][tgt])
        grown[tgt] = np.where(take, True, grown[tgt])
    f, dx1, dx2, dx1dx2, ok = (*out, grown)
    return f, dx1, dx2, dx1dx2, ok
