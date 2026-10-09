from .core import Interface
from .fluids.generic import (
    register_conductivity_residual,
    register_viscosity_higher_order,
)
from .utils.build_utils import tables

__all__ = [
    "Interface",
    "register_viscosity_higher_order",
    "register_conductivity_residual",
]
