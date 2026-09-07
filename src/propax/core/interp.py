import hashlib
from pathlib import Path
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from propax.utils.numerics import pick
from propax.utils.solvers import bisect
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

    __slots__ = ("array", "_hash", "shape")

    def __init__(self, a):
        a = np.ascontiguousarray(a)
        self.array = jnp.asarray(a)
        self.shape = jnp.asarray(a).shape
        self._hash = hash(
            (a.shape, a.dtype.str, hashlib.blake2b(a.data, digest_size=16).digest())
        )

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, _Const) and self._hash == other._hash

    def __repr__(self):
        return f"_Const(shape={self.array.shape}, dtype={self.array.dtype})"


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
    # polynomial roots with a sign change, define the the number of monotone segments
    cuts: _Const = eqx.field(static=True)  # (C, n_segments + 1)

    log_components: Tuple[int, ...] = eqx.field(static=True, default=())
    """Components stored as log(value), and exponentiated on the way out.

    A channel crossing decades -- rho_V spans ten of them, from 1e-8 at the
    triple point to rho_c -- cannot be certified on its value: `tail_ratio`
    weighs the last coefficients against the first, so it measures error
    against the channel's *largest* scale, and where the function is tiny an
    absolute error negligible there is a relative error of percents. On the
    logarithm the same certificate reads relative error at every point.
    """

    @classmethod
    def from_arrays(
        cls,
        edges,
        coeffs,
        cuts,
        log_components: Tuple[int, ...] = (),
        dtype=jnp.float64,
    ) -> "ChebyshevPieces":
        """Build from the stored layout.

        - `edges` is (n_pieces + 1,) in x the number of pieces is determined by
        an accuracy criterion by the dyadic splitting process
        - `coeffs` is (n_pieces, degree + 1, C)
        - `cuts` is (C, n_segments + 1), padded with the domain's own end where one
        component turns over less often than another.

        NOTE :
        The abscissa stays float64 whatever `dtype` the coefficients are stored at,
        this is essential for localization
        """
        return cls(
            edges=_Const(np.asarray(edges, dtype=np.float64)),
            coeffs=_Const(np.asarray(coeffs, dtype=dtype)),
            cuts=_Const(np.asarray(cuts, dtype=np.float64)),
            log_components=tuple(int(c) for c in log_components),
        )

    def __call__(self, x: jaxFloat) -> jax.Array:
        """Clenshaw on the piece holding x. Returns (C,)."""
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
        out = coeffs[i, 0] + t * b1 - b2
        if self.log_components:
            idx = jnp.asarray(self.log_components)
            out = out.at[idx].set(jnp.exp(out[idx]))
        return out

    def invert(self, y: jaxFloat, component: int = 0) -> jax.Array:
        """Every x with channel(x)[component] = y, one slot per monotone segment.

        Fixed shape, NaN where that segment does not reach y.
        """
        cuts = self.cuts.array[component]
        y = jnp.asarray(y)

        def on_segment(k):
            lo, hi = cuts[k], cuts[k + 1]
            root, converged = bisect(
                lambda x, _: self(x)[component] - y,
                lo,
                hi,
                None,
                max_steps=TOL.caps.bisect_steps,
                rtol=TOL.acc.sat_inversion_rtol,
                atol=TOL.acc.bisect_atol,
            )
            # a padded segment is a point: no bracket, hence no root
            return pick(converged & (hi > lo), root, jnp.nan)

        return jnp.stack([on_segment(k) for k in range(int(self.cuts.shape[1]))])


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
