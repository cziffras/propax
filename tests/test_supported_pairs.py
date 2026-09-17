"""
`fast_flash` reads a table where `flash` brackets, so the two can
disagree about which pairs work.
"""

from itertools import combinations

import pytest

from propax import Interface
from propax.core.config import TABLE_REGISTRY, ThermoVar
from propax.core.flash.dispatch import INVALID_PAIRS, check_supported

# a quality is read straight off the saturation curve, and (D, T) is the state
# already: neither reaches an interpolator
NO_TABLE_NEEDED = {"saturated", "natural"}
# the density jumps across P_sat(T), which no bicubic holds, and the flash is
# already a 1D solve there + this flash is almost as fast as a lookup in practice
UNTABLED = {frozenset({ThermoVar.P, ThermoVar.T})}

TABLED = {frozenset({c.x_axis.variable, c.y_axis.variable}) for c in TABLE_REGISTRY}


def _key(pair) -> frozenset:
    return frozenset(ThermoVar(v) for v in pair)


class TestSupportedPairs:
    def test_every_pair_is_either_supported_or_refused_with_a_reason(self):
        inputs = [v for v in ThermoVar if v.spec.can_be_input]
        supported = Interface.supported_pairs()
        for a, b in combinations(inputs, 2):
            named = tuple(sorted((a.value, b.value)))
            if named in supported:
                check_supported(a, b)
            else:
                assert frozenset({a, b}) in INVALID_PAIRS, (
                    f"({a.value}, {b.value}) is neither supported nor refused"
                )
                with pytest.raises(ValueError):
                    check_supported(a, b)

    def test_the_order_of_the_two_does_not_matter(self):
        for pair in Interface.supported_pairs():
            a, b = (ThermoVar(v) for v in pair)
            check_supported(a, b)
            check_supported(b, a)

    def test_routes_are_the_ones_the_dispatch_implements(self):
        assert set(Interface.supported_pairs().values()) <= {
            "saturated",
            "natural",
            "one_dim",
            "nested",
        }


class TestTableCoverage:
    def test_every_solvable_pair_has_a_table(self):
        missing = [
            pair
            for pair, route in Interface.supported_pairs().items()
            if route not in NO_TABLE_NEEDED and _key(pair) not in TABLED | UNTABLED
        ]
        assert not missing, f"no interpolation table for {missing}"

    def test_no_table_describes_a_pair_the_flash_refuses(self):
        supported = {_key(p) for p in Interface.supported_pairs()}
        orphans = [sorted(v.value for v in k) for k in TABLED if k not in supported]
        assert not orphans, f"tables for unsupported pairs: {orphans}"

    def test_each_table_stores_whichever_of_density_and_temperature_it_lacks(self):
        for cfg in TABLE_REGISTRY:
            axes = {cfg.x_axis.variable, cfg.y_axis.variable}
            assert set(cfg.outputs) == {ThermoVar.D, ThermoVar.T} - axes, cfg.name

    def test_table_names_are_unique(self):
        names = [c.name for c in TABLE_REGISTRY]
        assert len(names) == len(set(names))
