from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .precision import PRECISION

# ------------------------------------------------------------------------
# SUPERANCILLARIES MATERIAL : find all details in docs/superancillaries.md
# ------------------------------------------------------------------------


_TAIL = 3
"""Coefficients at each end whose norms form the convergence certificate."""

_MAX_PASSES = 12
"""Dyadic halvings allowed."""

_DEGREES = (8, 12, 16, 24, 32, 48)
"""Degrees each channel chooses from, minimising `pieces * (degree + 1)`."""

_IM_TOL = 1e-10
"""Imaginary part below which a colleague-matrix eigenvalue counts as real."""

_OUT_TOL = 1e-8
"""How far outside [-1, 1] a root may stray before it is discarded as spurious."""


def cheb_lobatto_nodes(degree: int, xmin: float, xmax: float) -> np.ndarray:
    t = np.cos(np.arange(degree + 1) * np.pi / degree)
    return 0.5 * ((xmax - xmin) * t + (xmax + xmin))


def _values_to_coeffs_matrix(degree: int) -> np.ndarray:
    n = degree
    j = np.arange(n + 1)
    p = np.ones(n + 1)
    p[0] = p[n] = 2.0
    k = j[:, None]
    return (2.0 / (n * p[:, None])) * np.cos(j[None, :] * k * np.pi / n) / p[None, :]


@dataclass(frozen=True)
class ChebyshevExpansion:
    """
    A truncated Chebyshev series, of `C` components sharing one interval to ensure
    coherence and avoid padding.
    """

    xmin: float
    xmax: float
    coeffs: np.ndarray

    def _to_unit(self, x):
        """Map [xmin, xmax] onto [-1, 1], where the polynomials are defined (see docs)."""
        return (2.0 * x - (self.xmax + self.xmin)) / (self.xmax - self.xmin)

    def __call__(self, x):
        """Clenshaw's recurrence that evaluates T_k without ever forming it.

        `x` of shape S and `coeffs` of shape (degree + 1, C) give S + (C,); a
        one-dimensional `coeffs` gives S, so the scalar case is unchanged.
        """
        t = self._to_unit(np.asarray(x, dtype=float))
        c = self.coeffs
        if c.ndim > 1:  # make room for the component axis
            t = t[..., None]
        b1 = np.zeros_like(t)
        b2 = np.zeros_like(t)
        for k in range(len(c) - 1, 0, -1):
            b1, b2 = 2.0 * t * b1 - b2 + c[k], b1
        return c[0] + t * b1 - b2

    def derivative(self) -> "ChebyshevExpansion":
        c = self.coeffs
        n = len(c) - 1
        d = np.zeros((max(n, 1),) + c.shape[1:])
        for k in range(n, 0, -1):
            d[k - 1] = (d[k + 1] if k + 1 < len(d) else 0.0) + 2.0 * k * c[k]
        d[0] *= 0.5
        return ChebyshevExpansion(
            self.xmin, self.xmax, d * (2.0 / (self.xmax - self.xmin))
        )

    def tail_ratio(self, tail: int = _TAIL) -> float:
        """Norm of the last `tail` coefficients over the first `tail`.

        Takes the worst component convergence : a piece must certify
        every quantity it carries.
        """
        c = np.atleast_2d(self.coeffs.T).T  # (degree + 1, C)
        head = np.linalg.norm(c[:tail], axis=0)
        tail_n = np.linalg.norm(c[-tail:], axis=0)
        ratio = np.where(head == 0.0, 0.0, tail_n / np.where(head == 0.0, 1.0, head))
        return float(np.max(ratio))

    def component(self, i: int) -> "ChebyshevExpansion":
        """One component on its own, for the scalar machinery below."""
        c = np.atleast_2d(self.coeffs.T).T
        return ChebyshevExpansion(self.xmin, self.xmax, c[:, i])

    def roots(self) -> np.ndarray:
        """Every real root in [xmin, xmax], via the colleague matrix."""
        # scalar only: the colleague matrix is one series' companion, so a
        # vector expansion is asked for its components one at a time
        c = np.trim_zeros(np.asarray(self.coeffs).ravel(), "b")
        n = len(c) - 1
        if n < 1:
            return np.empty(0)
        if n == 1:
            r = np.array([-c[0] / c[1]])
            return self._keep_inside(r)

        A = np.zeros((n, n))
        A[0, 1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = 0.5
            A[i, i + 1] = 0.5
        A[n - 1, n - 2] = 0.5
        A[n - 1, :] -= c[:n] / (2.0 * c[n])

        ev = np.linalg.eigvals(A)
        real = ev[np.abs(ev.imag) < _IM_TOL].real
        return self._keep_inside(real)

    def _keep_inside(self, unit_roots: np.ndarray) -> np.ndarray:
        # strictly inside: a root just outside must not be clamped onto the
        # edge, or every piece whose derivative merely comes close to zero at
        # its boundary reports a spurious extremum there
        keep = unit_roots[
            (unit_roots > -1.0 + _OUT_TOL) & (unit_roots < 1.0 - _OUT_TOL)
        ]
        x = 0.5 * ((self.xmax - self.xmin) * keep + (self.xmax + self.xmin))
        return np.sort(x)


def fit_expansion(
    f: Callable[[np.ndarray], np.ndarray], xmin: float, xmax: float, degree: int
) -> ChebyshevExpansion:
    """One expansion of `f` on [xmin, xmax], f evaluated at the Lobatto points."""
    x = cheb_lobatto_nodes(degree, xmin, xmax)
    y = np.asarray(f(x), dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError(f"non-finite sample on [{xmin}, {xmax}]")
    return ChebyshevExpansion(xmin, xmax, _values_to_coeffs_matrix(degree) @ y)


def dyadic_split(
    f: Callable[[np.ndarray], np.ndarray],
    xmin: float,
    xmax: float,
    degree: int,
    max_passes: int = _MAX_PASSES,
) -> List[ChebyshevExpansion]:
    """Cover [xmin, xmax] with expansions that each pass the tail test.

    Fit the whole interval; wherever the certificate fails, halve and refit.
    """
    pieces = [fit_expansion(f, xmin, xmax, degree)]
    for _ in range(max_passes):
        converged = True
        out: List[ChebyshevExpansion] = []
        for piece in pieces:
            if piece.tail_ratio(_TAIL) <= PRECISION.tol:
                out.append(piece)
                continue
            mid = 0.5 * (piece.xmin + piece.xmax)
            out.append(fit_expansion(f, piece.xmin, mid, degree))
            out.append(fit_expansion(f, mid, piece.xmax, degree))
            converged = False
        pieces = out
        if converged:
            break
    return pieces


# ----------------------------------------------------------------------------
# One channel --> piecewise expansion, cut into monotone segments for accuracy
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MonotoneSegment:
    xmin: float
    xmax: float
    lo: float
    hi: float
    increasing: bool

    def contains(self, y: float) -> bool:
        return self.lo <= y <= self.hi


class ChebyshevChannel:
    """One quantity along the curve: evaluate it, or invert it.
    This is the `build side` module that does not require any JAX
    machinery."""

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        xmin: float,
        xmax: float,
        degree: Optional[int] = None,
    ):
        self.xmin, self.xmax = xmin, xmax
        if degree is None:
            fits = {d: dyadic_split(f, xmin, xmax, d) for d in _DEGREES}
            # degree trades against piece count, while maintaining accuracy we aim
            # to minimize storage use since all parameters are to be loaded in RAM
            # at runtime
            degree = min(fits, key=lambda d: len(fits[d]) * (d + 1))
            self.pieces = fits[degree]
        else:
            self.pieces = dyadic_split(f, xmin, xmax, degree)
        self.degree = degree
        self.segments = self._build_segments()

    def _piece_of(self, x: float) -> ChebyshevExpansion:
        for piece in self.pieces:
            if piece.xmin <= x <= piece.xmax:
                return piece
        return self.pieces[0] if x < self.xmin else self.pieces[-1]

    def __call__(self, x):
        scalar = np.ndim(x) == 0
        xs = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.stack([np.asarray(self._piece_of(v)(v)) for v in xs])
        return out[0] if scalar else out

    @property
    def n_components(self) -> int:
        c = self.pieces[0].coeffs
        return 1 if c.ndim == 1 else c.shape[1]

    def _derivative_at(self, x: float, i: int) -> float:
        p = self._piece_of(x)
        return float(p.component(i).derivative()(x))

    def _interior_extrema(self, i: int) -> np.ndarray:
        """Where component `i`'s derivative vanishes, across all the pieces.

        Per component: two quantities fitted on one interval have no reason to
        turn over at the same place, and each is inverted on its own.
        """
        found = [p.component(i).derivative().roots() for p in self.pieces]
        x = np.concatenate(found) if found else np.empty(0)
        span = self.xmax - self.xmin
        # take the interior points
        x = np.sort(x[(x > self.xmin + 1e-12) & (x < self.xmax - 1e-12)])
        if x.size == 0:
            return x
        # merge points that are extremely close on the axis; 1e-9 because
        # when evaluating the sign of the slope we use the below eps that
        # needs to be greater than this tolerance
        x = x[np.concatenate([[True], np.diff(x) > 1e-9 * span])]

        eps = 1e-7 * span
        turning = [
            v
            for v in x
            if self._derivative_at(max(self.xmin, v - eps), i)
            * self._derivative_at(min(self.xmax, v + eps), i)
            < 0.0
        ]  # determine the sign of the slope at each of those points
        return np.asarray(turning)

    def _build_segments(self) -> List[List[MonotoneSegment]]:
        """One list of monotone stretches per component."""
        out = []
        for i in range(self.n_components):
            cuts = np.concatenate([[self.xmin], self._interior_extrema(i), [self.xmax]])
            segs = []
            for a, b in zip(cuts[:-1], cuts[1:]):
                ya = float(np.atleast_1d(self(a))[i])
                yb = float(np.atleast_1d(self(b))[i])
                segs.append(
                    MonotoneSegment(
                        xmin=float(a),
                        xmax=float(b),
                        lo=min(ya, yb),
                        hi=max(ya, yb),
                        increasing=yb > ya,
                    )
                )
            out.append(segs)
        return out

    def __repr__(self) -> str:
        worst = max(g.tail_ratio(_TAIL) for g in self.pieces)
        return (
            f"<ChebyshevChannel degree {self.degree}, {len(self.pieces)} pieces, "
            f"{[len(g) for g in self.segments]} monotone segment(s), "
            f"worst tail {worst:.1e}>"
        )
