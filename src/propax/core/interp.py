import hashlib
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from propax.utils.numerics import pick, safe_div
from propax.utils.solvers import newton_loop
from propax.utils.types import jaxFloat

from .tolerances import TOL


def hermite_weights(t):

    t2 = t * t
    t3 = t2 * t
    return jnp.stack(
        [
            1.0 - 3.0 * t2 + 2.0 * t3,
            3.0 * t2 - 2.0 * t3,
            t - 2.0 * t2 + t3,
            t3 - t2,
        ]
    )


def _is_uniform(arr: np.ndarray) -> bool:
    """Whether an axis is evenly spaced, to within its own rounding.

    Decides between a division and a binary search in `BicubicInterpolation.__call__`,
    so it has to be exact about what the localization can actually assume: the
    fast path computes `floor((x - x0) / h)`, whose worst-case index drift over
    the whole axis is what the tolerance is scaled against. A grid built as a
    linspace is uniform to a few ulps and passes; a graded one fails by orders
    of magnitude, so nothing sits near the threshold in practice.
    """
    if arr.shape[0] < 3:
        return True
    d = np.diff(np.asarray(arr, dtype=np.float64))
    return bool(np.max(np.abs(d - d[0])) <= 1e-9 * abs(d[0]))


class _Const:
    """
    An array held as a static equinox field, becomes a trace-time constant
    preventing any batching in vmaps duplicating it, that might cause OOM.
    """

    __slots__ = ("_host", "_device", "_hash", "shape")

    def __init__(self, a):
        a = np.ascontiguousarray(a)
        # avoid desynchronizing host and device
        # (host stores a new value if the user
        # sets it and device did not modify its cache)
        a.flags.writeable = False
        self._host = a
        self._device = {}
        self.shape = a.shape
        self._hash = hash((a.shape, a.dtype.str, hashlib.blake2b(a.data).digest()))

    @property
    def array(self):
        x64 = jax.config.read("jax_enable_x64")
        if x64 not in self._device:
            # do not cache a tracer
            with jax.ensure_compile_time_eval():
                self._device[x64] = jnp.asarray(self._host)
        return self._device[x64]

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, _Const) and self._hash == other._hash

    def __repr__(self):
        return f"_Const(shape={self._host.shape}, dtype={self._host.dtype})"


class ChebyshevPieces(eqx.Module):
    """A piecewise Chebyshev channel, evaluated. Built by `utils.exact.superancillary`.

    Comments below are mainly detailing JAX-specific implementation things, for theoretical
    explanation see docs/superancillaries.md where all formula or proof helping understanding
    is provided in clear form.
    """

    edges: _Const = eqx.field(
        static=True
    )  # (n_pieces + 1,) piece boundaries determined through dydadic splitting
    coeffs: _Const = eqx.field(static=True)  # (n_pieces, degree + 1)

    log_components: Tuple[int, ...] = eqx.field(static=True, default=())
    """Components stored as log(value), and exponentiated on the way out.
    """

    @classmethod
    def from_arrays(
        cls,
        edges,
        coeffs,
        log_components: Tuple[int, ...] = (),
        dtype=jnp.float64,
    ) -> "ChebyshevPieces":
        """Build from the stored layout.

        - `edges` is (n_pieces + 1,) in x the number of pieces is determined by
        an accuracy criterion by the dyadic splitting process
        - `coeffs` is (n_pieces, degree + 1, C)

        NOTE :
        The abscissa stays float64 whatever `dtype` the coefficients are stored at,
        this is essential for localization
        """
        return cls(
            edges=_Const(np.asarray(edges, dtype=np.float64)),
            coeffs=_Const(np.asarray(coeffs, dtype=dtype)),
            log_components=tuple(int(c) for c in log_components),
        )

    def __call__(self, x: jaxFloat) -> jax.Array:
        """The channel at x. Returns (C,)."""
        out = self._stored(x)
        if self.log_components:
            idx = jnp.asarray(self.log_components)
            out = out.at[idx].set(jnp.exp(out[idx]))
        return out

    def _stored(self, x: jaxFloat) -> jax.Array:
        """Clenshaw iteration on the piece holding x, log components left as logs. Returns (C,)."""
        edges, coeffs = self.edges.array, self.coeffs.array
        x = jnp.asarray(x)
        i = jnp.clip(
            jnp.searchsorted(edges, x, side="right") - 1, 0, coeffs.shape[0] - 1
        )
        lo, hi = edges[i], edges[i + 1]
        t = (2.0 * x - (hi + lo)) / (hi - lo)

        # Used to be c = coeffs[i] and then looping as below, however materialising
        # c[i] as a (N, degree + 1, C) array in vmap multiplies 5x execution time
        # ~25ms instead of ~5ms
        b1 = b2 = jnp.zeros_like(coeffs[0, 0])
        for k in range(coeffs.shape[1] - 1, 0, -1):
            b1, b2 = 2.0 * t * b1 - b2 + coeffs[i, k], b1
        return coeffs[i, 0] + t * b1 - b2

    def invert(self, value: jaxFloat, component: int | jax.Array = 0) -> jax.Array:
        """
        Every stored component (rho_L, rho_V, ln P_sat) is monotone in T, so the
        whole channel is one bracket. `component` may be traced; a log component
        is inverted on its logarithm, where it is stored.
        """
        return _abscissa_of(self, component, jnp.asarray(value))


def _as_stored(channel: ChebyshevPieces, component, value):
    """`value` as the component is stored: its logarithm for a log component."""
    is_log = jnp.isin(component, jnp.asarray(channel.log_components, dtype=int))
    return pick(is_log, jnp.log(pick(is_log, value, 1.0)), value)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _abscissa_of(channel: ChebyshevPieces, component, value):
    x_lo, x_hi = channel.edges.array[0], channel.edges.array[-1]
    target = _as_stored(channel, component, value)

    def stored(x):
        return channel._stored(x)[component]

    def residual(x, aim):
        stored_x, slope = jax.jvp(stored, (x,), (jnp.ones_like(x),))
        return stored_x - aim, slope

    at_lo, at_hi = stored(x_lo), stored(x_hi)
    low, high = jnp.minimum(at_lo, at_hi), jnp.maximum(at_lo, at_hi)
    edge = TOL.acc.channel_edge * jnp.maximum(
        jnp.maximum(jnp.abs(low), jnp.abs(high)), 1.0
    )
    in_range = (target >= low - edge) & (target <= high + edge)
    held = jnp.clip(target, low, high)
    chord = jnp.clip(safe_div(held - at_lo, at_hi - at_lo), 0.0, 1.0)
    seed = x_lo + chord * (x_hi - x_lo)
    # a lane the channel does not reach is handed the value at its seed,
    # so it stops at once instead of running every step of the batch
    aim = pick(in_range, held, stored(seed))
    x, _ = newton_loop(
        residual,
        x_lo,
        x_hi,
        aim,
        x0=seed,
        max_steps=TOL.caps.newton_steps,
        rtol=TOL.acc.newton_rtol,
    )
    return pick(in_range, x, jnp.nan)


@_abscissa_of.defjvp
def _abscissa_of_jvp(channel: ChebyshevPieces, primals, tangents):
    component, value = primals
    _, value_dot = tangents
    x = _abscissa_of(channel, component, value)
    found = jnp.isfinite(x)
    # implicit function theorem: dx = d(stored value) / (d stored / dx),
    # the numerator carrying the 1/value of a log component
    _, target_dot = jax.jvp(
        lambda v: _as_stored(channel, component, v), (value,), (value_dot,)
    )
    at = pick(found, x, channel.edges.array[0])
    slope = jax.grad(lambda x_: channel._stored(x_)[component])(at)
    return x, pick(found, safe_div(target_dot, slope), 0.0)


class BicubicInterpolation(eqx.Module):
    """A cell-wise bicubic over a rectangular grid, with C channels.

    Each node carries the value and the three derivatives a bicubic needs,
    and a lookup weighs the four corners of the cell holding the point.
    """

    grid_x: _Const = eqx.field(static=True)
    grid_y: _Const = eqx.field(static=True)

    values: _Const = eqx.field(static=True)
    slope_x: _Const = eqx.field(static=True)
    slope_y: _Const = eqx.field(static=True)
    cross: _Const = eqx.field(static=True)

    log_x: bool = eqx.field(static=True)
    log_y: bool = eqx.field(static=True)

    step_x: float = eqx.field(static=True)
    step_y: float = eqx.field(static=True)

    x_name: str = eqx.field(static=True)
    y_name: str = eqx.field(static=True)

    output_names: Tuple[str, ...] = eqx.field(static=True)

    even_x: bool = eqx.field(static=True, default=True)
    even_y: bool = eqx.field(static=True, default=True)

    # nodes with no physical state, carrying a nearest-neighbour fill
    unreachable: Optional[_Const] = eqx.field(static=True, default=None)

    @classmethod
    def create(cls, table_path: str, dtype=jnp.float32):
        """
        Load a table from disk.

        `dtype` controls the storage precision of the four (Nx, Ny, C) data
        arrays (the grid axes always stay float64 for accurate localization).
        The default float32 halves device memory.
        """
        path = Path(table_path)
        if path.is_dir():
            path = path / "table_data.npz"

        data = np.load(path)
        axis_x, axis_y = data["arr_x1"], data["arr_x2"]

        channels = [data[k] for k in ("f", "dx1", "dx2", "dx1dx2")]
        if channels[0].ndim == 2:
            channels = [c[..., None] for c in channels]
        stored = np.dtype(jnp.dtype(dtype).name)
        values, slope_x, slope_y, cross = (
            _Const(np.asarray(c, dtype=stored)) for c in channels
        )

        blank = data["unreachable"] if "unreachable" in data.files else None

        return cls(
            grid_x=_Const(axis_x),
            grid_y=_Const(axis_y),
            values=values,
            slope_x=slope_x,
            slope_y=slope_y,
            cross=cross,
            log_x=bool(data.get("is_log_x1", axis_x[0] > 0)),
            log_y=bool(data.get("is_log_x2", axis_y[0] > 0)),
            step_x=float(axis_x[1] - axis_x[0]),
            step_y=float(axis_y[1] - axis_y[0]),
            output_names=tuple(str(name) for name in data["output_names"]),
            x_name=str(data.get("x_name", "unknown")),
            y_name=str(data.get("y_name", "unknown")),
            even_x=_is_uniform(axis_x),
            even_y=_is_uniform(axis_y),
            unreachable=None if blank is None else _Const(blank),
        )

    def _locate(self, x, axis, even, step):
        """The index of the cell holding `x`, and that cell's width."""
        if even:
            k = jnp.floor((x - axis[0]) / step).astype(jnp.int32)
            k = jnp.clip(k, 0, axis.shape[0] - 2)
            return k, step
        k = jnp.clip(jnp.searchsorted(axis, x, side="right") - 1, 0, axis.shape[0] - 2)
        return k, axis[k + 1] - axis[k]

    def __call__(self, x_val: jaxFloat, y_val: jaxFloat) -> jax.Array:
        # materialize the static numpy tables as trace-time constants; raw numpy
        # cannot be indexed by a traced index (it would call __array__ on it)
        axis_x, axis_y = self.grid_x.array, self.grid_y.array

        x = jnp.asarray(x_val).reshape(())
        y = jnp.asarray(y_val).reshape(())
        x = jnp.log(x) if self.log_x else x
        y = jnp.log(y) if self.log_y else y

        # Outside the axes the clip below evaluates the edge cell and returns a
        # perfectly plausible number for a state the table never held
        outside = (
            (x < axis_x[0]) | (x > axis_x[-1]) | (y < axis_y[0]) | (y > axis_y[-1])
        )
        x = jnp.clip(x, axis_x[0], axis_x[-1])
        y = jnp.clip(y, axis_y[0], axis_y[-1])

        i, width_x = self._locate(x, axis_x, self.even_x, self.step_x)
        j, width_y = self._locate(y, axis_y, self.even_y, self.step_y)

        corners = jnp.ix_(jnp.array([i, i + 1]).ravel(), jnp.array([j, j + 1]).ravel())

        # stored derivatives are per internal unit, a cell wants them per cell:
        # hence the local width, and not a global step
        stored = self.values.array.dtype
        wx = jnp.asarray(width_x, dtype=stored)
        wy = jnp.asarray(width_y, dtype=stored)
        f = self.values.array[corners]
        fx = self.slope_x.array[corners] * wx
        fy = self.slope_y.array[corners] * wy
        fxy = self.cross.array[corners] * wx * wy

        # (4, 4, C): rows index [value@x0, value@x1, slope@x0, slope@x1] and
        # columns the same along y, which is the order `hermite_weights` returns
        block = jnp.stack(
            [
                jnp.stack([f[0, 0], f[0, 1], fy[0, 0], fy[0, 1]]),
                jnp.stack([f[1, 0], f[1, 1], fy[1, 0], fy[1, 1]]),
                jnp.stack([fx[0, 0], fx[0, 1], fxy[0, 0], fxy[0, 1]]),
                jnp.stack([fx[1, 0], fx[1, 1], fxy[1, 0], fxy[1, 1]]),
            ]
        )

        wu = hermite_weights((x - axis_x[i]) / width_x)
        wv = hermite_weights((y - axis_y[j]) / width_y)
        out = jnp.einsum("i,ijc,j->c", wu, block, wv)

        # a cell touching a node with no physical state rests on a fill value;
        # NaN says so instead of returning a plausible number and the solvers seed
        # from a mid-domain fallback when they see it
        if self.unreachable is not None:
            out = jnp.where(jnp.any(self.unreachable.array[corners]), jnp.nan, out)
        return jnp.where(outside, jnp.nan, out)
