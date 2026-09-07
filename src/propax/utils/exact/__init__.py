"""All exact computations contributing to the superancillaries construction,
the module is named `exact` after the fact it uses mpmath for high precision
computations which is required near the critical point.
"""

from .constants import critical_constants, density_ceiling, provisional_eos
from .critical import (
    CriticalPoint,
    saturated_pair,
    saturated_pairs,
    solve_critical_point,
)
from .curve import SaturationCurve, saturation_walk
from .fit import SaturationFit, fit_saturation, from_eos
from .superancillary import ChebyshevChannel

__all__ = [
    "CriticalPoint",
    "SaturationCurve",
    "ChebyshevChannel",
    "SaturationFit",
    "fit_saturation",
    "saturated_pair",
    "saturated_pairs",
    "critical_constants",
    "density_ceiling",
    "provisional_eos",
    "from_eos",
    "saturation_walk",
    "solve_critical_point",
]
