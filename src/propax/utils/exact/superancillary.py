from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from .precision import PRECISION

# -------------------------------------------------------------------------
# SUPERANCILLARIES MATERIAL : find all details in docs/superancillaries.md
#
# NOTE : for simplicity all fitted variables are monotone in T (Psat, rho_L,
#   rho_V) while we could have fitted calorific variables too as planned
#   initially, we realized it offered no more speed than reading the EOS
#   evaluated in rho_L, rho_V while complicating the code significantly
# ------------------------------------------------------------------------


_TAIL = 3
"""Coefficients at each end whose norms form the convergence certificate."""

_MAX_PASSES = 12
"""Dyadic halvings allowed."""

_DEGREES = (8, 16, 32)
"""Degrees each channel chooses from, minimising `pieces * (degree + 1)`."""


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
# One channel --> piecewise expansion
# ----------------------------------------------------------------------------


class ChebyshevChannel:
    """One quantity along the curve, fitted piecewise. This is the `build side`
    module that does not require any JAX machinery."""

    def __init__(self, f: Callable[[np.ndarray], np.ndarray], xmin: float, xmax: float):
        self.xmin, self.xmax = xmin, xmax
        fits = {d: dyadic_split(f, xmin, xmax, d) for d in _DEGREES}
        # degree trades against piece count, while maintaining accuracy we aim
        # to minimize storage use since all parameters are to be loaded in RAM
        # at runtime
        self.degree = min(fits, key=lambda d: len(fits[d]) * (d + 1))
        self.pieces = fits[self.degree]

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

    def __repr__(self) -> str:
        worst = max(g.tail_ratio(_TAIL) for g in self.pieces)
        return (
            f"<ChebyshevChannel degree {self.degree}, {len(self.pieces)} pieces, "
            f"worst tail {worst:.1e}>"
        )
