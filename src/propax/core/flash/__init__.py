"""The flash: given two properties, find the state.

`Interface` is the facade; the work is here, split by responsibility
rather than by size. Each module takes what it needs as arguments  an EOS, a
saturation module, a table  so a signature says which parts of the interface
a solver actually touches. Nothing here reaches back into the interface.

    dispatch       which strategy a pair takes, and nothing else
    two_phase      the lever rule, and the three ways T_sat is obtained
    single_phase   the bracketed 1D solve and the nested one
    results        turning a solved state into the returned dictionary
    tables         the optional interpolation accelerator, isolated
"""

from .dispatch import (
    check_supported,
    is_degenerate_pair,
    is_saturated_pair,
    supported_pairs,
)
from .results import add_transport, fill_result_dict, mixture_state
from .tables import call_interp

__all__ = [
    "is_saturated_pair",
    "check_supported",
    "is_degenerate_pair",
    "supported_pairs",
    "add_transport",
    "fill_result_dict",
    "mixture_state",
    "call_interp",
]
