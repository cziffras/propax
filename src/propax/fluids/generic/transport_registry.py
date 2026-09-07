"""
User-supplied transport terms.

For a literature or hardcoded correlation that propax has no built-in slot class
for (a simple example would be hydrogen's Muzny higher-order) the user registers
a JAX term here and references it from the fluid file with a slot :

    "higher_order": {"type": "custom", "name": "<name_I_gave>", "params": {...}}

A registered builder receives the slot `params` (a plain dict from the file)
and a context and returns an eqx.Module exposing the slot's method :

    viscosity higher-order  ->  contribution(rho_molar, T, eta0) -> Pa.s
    conductivity residual   ->  contribution(rho, T)             -> W/(m K)

The context carries the EOS (and, for conductivity, the viscosity), so
EOS-coupled terms can be expressed easily.

From the user point a view, here are the lines necessary to add a new term::

    class MyTerm(
        eqx.Module
    ):
        a: float
        b: float
        c: float

        def contribution(
            self,
            rho,
            T,
            eta0,
        ):  # in visc case
            return (
                self.a
                * T
                + self.b
                * rho
                + self.c
                * eta0
            )


    register_viscosity_higher_order(
        "myterm", MyTerm
    )

Obviously the term's parameters params have to be added to the fluid's json
under the `myterm` sub-key, `Interface.create` deals with the rest !
"""

import inspect
from typing import Callable, Dict

import equinox as eqx

_VISCOSITY_HIGHER_ORDER: Dict[str, Callable] = {}
_CONDUCTIVITY_RESIDUAL: Dict[str, Callable] = {}


def _register(registry, name):

    def deco(obj):
        if not callable(obj):
            raise TypeError(
                f"cannot register {obj!r} under '{name}': expected an eqx.Module "
                "subclass, or a builder taking (params, *, eos[, viscosity])"
            )
        registry[name] = obj
        return obj

    return deco


def _instantiate(entry, name, kind, params: dict, **context):
    term = entry(**params) if inspect.isclass(entry) else entry(params, **context)
    if not isinstance(term, eqx.Module) or not callable(
        getattr(term, "contribution", None)
    ):
        raise TypeError(
            f"the custom {kind} term '{name}' produced {term!r}, which is not an "
            "eqx.Module carrying a `contribution` method"
        )
    return term


def register_viscosity_higher_order(name: str, builder: Callable | None = None):
    deco = _register(_VISCOSITY_HIGHER_ORDER, name)
    if builder is None:
        return deco  # Used as a decorator: @register_viscosity_higher_order("name")
    return deco(
        builder
    )  # Used as a function: register_viscosity_higher_order("name", MyClass)


def register_conductivity_residual(name: str, builder: Callable | None = None):
    deco = _register(_CONDUCTIVITY_RESIDUAL, name)
    if builder is None:
        return deco
    return deco(builder)


_SIGNATURE = {
    "viscosity higher order": "contribution(self, rho_molar, T, eta0) -> Pa.s",
    "conductivity residual": "contribution(self, rho, T) -> W/(m K)",
}


def _lookup(registry, kind, name):
    if name not in registry:
        register = f"register_{kind.replace(' ', '_')}"
        raise KeyError(
            f"the fluid file asks for the custom {kind} term '{name}', which "
            f"nothing has registered (known: {sorted(registry) or 'none'}). "
            f"Define it and register it before Interface.create:\n\n"
            f"    import equinox as eqx\n"
            f"    from propax import {register}\n\n"
            f"    class MyTerm(eqx.Module):\n"
            f"        a: float\n\n"
            f"        def {_SIGNATURE[kind]}:\n"
            f"            ...\n\n"
            f'    {register}("{name}", MyTerm)\n\n'
            f"MyTerm is built as MyTerm(**params) from that slot's `params`, or "
            f"as builder(params, *, eos[, viscosity]) if you register a function."
        )
    return registry[name]


def build_viscosity_higher_order(name: str, params: dict, *, eos):
    kind = "viscosity higher order"
    entry = _lookup(_VISCOSITY_HIGHER_ORDER, kind, name)
    return _instantiate(entry, name, kind, params, eos=eos)


def build_conductivity_residual(name: str, params: dict, *, eos, viscosity):
    kind = "conductivity residual"
    entry = _lookup(_CONDUCTIVITY_RESIDUAL, kind, name)
    return _instantiate(entry, name, kind, params, eos=eos, viscosity=viscosity)
