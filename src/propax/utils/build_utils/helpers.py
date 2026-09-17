import logging

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from ...core.config import get_chunk_size  # noqa: E402
from ...core.saturation import Superancillary  # noqa: E402
from ...fluids._registry import EQS_REGISTRY  # noqa: E402

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _get_fluid_modules(fluid_name: str):
    processed = fluid_name.lower().replace(" ", "")
    if processed not in EQS_REGISTRY:
        available = ", ".join(sorted(EQS_REGISTRY.keys()))
        raise ValueError(
            f"Fluid '{fluid_name}' not found in registry. Available fluids: {available}"
        )
    eos = EQS_REGISTRY[processed][0]()
    return processed, eos, Superancillary.create(eos, processed)


def _mapped(fn, *arrays):
    """`fn` over every element of `arrays`, at most PROPAX_CHUNK_SIZE at a time."""
    xs = tuple(jnp.asarray(a) for a in arrays)
    chunk = get_chunk_size()
    if chunk is None:
        out = jax.jit(jax.vmap(fn))(*xs)
    else:
        mapped = lambda ts: jax.lax.map(lambda t: fn(*t), ts, batch_size=chunk)  # noqa: E731
        out = jax.jit(mapped)(xs)
    return jax.tree_util.tree_map(np.asarray, out)


def fill_ghost(arr_x1, arr_x2, f, dx1, dx2, dx1dx2, ok):
    """Grow the known region by one node in every direction (including diagonal steps),
    by a Taylor step.

    THIS IS AN EXTRAPOLATION: called until the grid is full, it keeps the table
    free of NaN with reasonably erroneous values. The caller keeps its own mask
    of the solved nodes, the ghosts are never counted as such.

    Fields are (n1, n2, channels), the mask `ok` is (n1, n2).
    """
    f, dx1, dx2, dx1dx2, ok = (a.copy() for a in (f, dx1, dx2, dx1dx2, ok))
    h1 = np.diff(arr_x1)[:, None, None]
    h2 = np.diff(arr_x2)[None, :, None]
    lo, hi, all_ = slice(0, -1), slice(1, None), slice(None)

    # one entry per direction: target nodes, their source nodes, and the step
    # from source to target along x1 and x2
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
        # a node is filled once, from a node known before this call
        take = ~ok[tgt] & ok[src] & ~grown[tgt]
        if not take.any():
            continue

        m = take[..., None]
        out[0][tgt] = np.where(m, f[src] + dx1[src] * sx + dx2[src] * sy, out[0][tgt])
        out[1][tgt] = np.where(m, dx1[src] + dx1dx2[src] * sy, out[1][tgt])
        out[2][tgt] = np.where(m, dx2[src] + dx1dx2[src] * sx, out[2][tgt])
        out[3][tgt] = np.where(m, dx1dx2[src], out[3][tgt])
        grown[tgt] = np.where(take, True, grown[tgt])
    return (*out, grown)
