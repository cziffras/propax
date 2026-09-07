"""
Bicubic interpolation are used for :

  1 - computing proposals for solvers (reduced precision)
  2 - computing full solutions in fast solve mode (see `fast_flash` in `Interface`)

For ensuring coherence, all tables are computed without calling
CoolProp using only jax-native HEOS, we split as follows :

  - `helpers`    : shared foundation (fluid modules, chunked vmap, constants)
  - `single_phase`: single-phase nodal values, on the runtime's own solvers
  - `dome`       : two-phase (saturation-dome) node values
  - `assemble`   : orchestration --> builds and writes each table
"""

from ...core.config import get_table_path
from ...fluids._registry import EQS_REGISTRY
from .assemble import (
    build_adaptive_table,
    build_vectorized_table,
    fluid_table_registry,
)
from .helpers import logger

__all__ = [
    "get_table_path",
    "EQS_REGISTRY",
    "build_adaptive_table",
    "logger",
    "build_vectorized_table",
    "fluid_table_registry",
]
