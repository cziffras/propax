from .core import Interface
from .fluids.generic import (
    register_conductivity_residual,
    register_viscosity_higher_order,
)

# restrain what can be imported if user enters `from propax import *`
__all__ = [
    "Interface",
    "register_viscosity_higher_order",
    "register_conductivity_residual",
]
