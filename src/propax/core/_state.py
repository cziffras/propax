from collections.abc import Mapping
from typing import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from propax.utils.types import jaxBool

from .config import ThermoVar


class PropertyMap(Mapping):
    """A property map that cannot be mutated in place.

    Built from `ThermoVar` keys, read back by either those or the internal
    name they carry. `replace` returns a new map, so a value
    reaching a traced call is never rebound behind it.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries=()):
        stored = {}
        for key, value in dict(entries).items():
            if not isinstance(key, ThermoVar):
                raise KeyError(
                    f"{key!r} cannot key a PropertyMap, only a ThermoVar can. "
                    "Use PropertyMap.of_internal to build one from internal names."
                )
            stored[key.internal_key] = value
        self._entries = stored

    @classmethod
    def of_internal(cls, entries) -> "PropertyMap":
        built = cls.__new__(cls)
        built._entries = dict(entries)
        return built

    def replace(self, entries) -> "PropertyMap":
        merged = dict(self._entries)
        for key, value in dict(entries).items():
            merged[key.internal_key if isinstance(key, ThermoVar) else key] = value
        return PropertyMap.of_internal(merged)

    def __getitem__(self, key):
        name = key.internal_key if isinstance(key, ThermoVar) else key
        try:
            return self._entries[name]
        except KeyError:
            held = ", ".join(sorted(self._entries)) or "nothing"
            raise KeyError(f"{key!r} is not in this map, which holds {held}") from None

    def __repr__(self) -> str:
        return f"PropertyMap({', '.join(sorted(self._entries))})"

    # functions to implement for mapping
    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __reduce__(self):
        names, values = _split(self)
        return (_rebuild, (names, values))


def _split(state: PropertyMap):
    names = tuple(sorted(state))
    return names, tuple(state[name] for name in names)


def _rebuild(names, values) -> PropertyMap:
    return PropertyMap.of_internal(zip(names, values))


def _flatten(state: PropertyMap):
    names, values = _split(state)
    return values, names


jax.tree_util.register_pytree_node(PropertyMap, _flatten, _rebuild)


def _to_array(v):
    return jnp.asarray(v)


class SaturationResult(eqx.Module):
    """A point of the saturation line: T and P, and D, U, H, S on both
    branches, with no quality to place the state between them."""

    L: PropertyMap
    V: PropertyMap
    T: Array = eqx.field(converter=_to_array)
    P: Array = eqx.field(converter=_to_array)
    is_valid: jaxBool = eqx.field(default=True, converter=_to_array)
