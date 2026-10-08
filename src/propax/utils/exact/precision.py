from dataclasses import dataclass

import mpmath as mp

# NOTE : offline mode is ALWAYS ran in full precision on CPU


@dataclass(frozen=True)
class Precision:
    tol: float = 1e-13
    """What a Chebyshev piece must certify before the splitting accepts it.
    Every precision below follows from it."""

    dps: int = 50
    """Working precision for the critical point and the saturated pairs. Its
    residuals reach 1e-50 by default."""

    newton_headroom: int = 8
    """Digits between the working precision and where a Newton is asked to
    stop. Converging to the last carried digit would ask the iteration to
    resolve its own round-off."""

    guard_slope: float = 1.5
    """Digits of significance the equilibrium residual cancels per decade of
    theta. Derived in superancillary.md."""

    guard_floor: int = 10
    """Guard digits carried whatever theta is, for the round-off that has
    nothing to do with the critical point."""

    def newton_tol(self, dps: int) -> mp.mpf:
        return mp.mpf(10) ** (-(dps - self.newton_headroom))

    def guard_digits(self, theta) -> int:
        th = mp.mpf(theta)
        if th <= 0:
            return self.guard_floor
        return int(mp.ceil(mp.mpf(self.guard_slope) * -mp.log10(th))) + self.guard_floor

    def working_dps(self, theta) -> int:
        return self.dps + self.guard_digits(theta)


PRECISION = Precision()
