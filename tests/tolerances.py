from dataclasses import dataclass, field

from propax.core.tolerances import _SCALING_FACTOR, TOL, PrecisionScaled


@dataclass(frozen=True)
class Transcription(PrecisionScaled):
    """Whether the offline pipeline produced the correlation it claims to.

    Tight tolerance against CoolProp.
    """

    eos_transcription: float = field(default=1e-9, metadata={"scales": False})

    state_from_rho_T: float = field(default=1e-9, metadata={"loosest": 1e-6})
    derivative: float = field(default=1e-9, metadata={"loosest": 1e-5})

    criticality_first: float = field(default=1e-9, metadata={"loosest": 1e-3})
    criticality_second: float = field(default=1e-6, metadata={"loosest": 1e-2})
    """dP/drho and d2P/drho2 at the solved critical point, scaled by the
    pressure's own size there, so this reads `flat to the last digits`."""

    critical_point_vs_source: float = field(default=1e-9, metadata={"scales": False})


@dataclass(frozen=True)
class RightRoot(PrecisionScaled):
    """Whether the root propax converged to is the physical one."""

    temperature: float = field(default=1e-5, metadata={"loosest": 1e-3})
    density: float = field(default=1e-5, metadata={"loosest": 1e-3})
    energy: float = field(default=1e-5, metadata={"loosest": 1e-3})

    phase_disagreement: float = field(default=0.05, metadata={"scales": False})
    oracle_fraction: float = field(default=0.55, metadata={"scales": False})


@dataclass(frozen=True)
class Residual(PrecisionScaled):
    """What the flash owes on its own equation, with no oracle involved."""

    slack: float = field(default=10.0, metadata={"scales": False})
    solved_fraction: float = field(default=0.99, metadata={"scales": False})

    saturation_inverts: float = field(default=1e-10, metadata={"loosest": 1e-3})
    equal_fugacity: float = field(default=1e-10, metadata={"loosest": 1e-3})

    @property
    def bound(self) -> float:
        return self.slack * TOL.acc.root_atol


def for_precision(x64: bool):
    factor = 1.0 if x64 else _SCALING_FACTOR
    return (
        Transcription().scaled(factor),
        RightRoot().scaled(factor),
        Residual().scaled(factor),
    )
