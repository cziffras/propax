"""
The oracle tests run against a seeded random sample
of whatever ships in `fluids/data`.

Precision is chosen once per process, by `PROPAX_TEST_X64`, and the whole suite
runs in it -- toggling `jax_enable_x64` mid-session would leave fixtures built
in one format and read in another. Run it twice to cover both:

    pytest                        # float64, the default
    PROPAX_TEST_X64=0 pytest      # float32, bounds widened by 2**29


==================================== TEST STRATEGY ====================================

Testing propax is tough, choosing tolerances across fluids, knowing wether it is
CoolProp or propax that is failing when testing against an oracle... We chose to
proceed as follows

- 1. sample a rhoT grid (native EOS) from propax within fluid specific bounds
- 2. compare this grid with the the native EOS implemented in CoolProp with a tight tolerance
    --> no solver comes into this step : propax and CoolProp might disagree on their tolerances
- 3. solve all flashes on each point of the grid with propax and check the residuals with a harsh
    tolerance
- 4. finally compare against the oracle (CoolProp) with a looser tolerance to check if propax did
    not find a metastable state (that could cancel the residual without solving the problem)
"""

import os

import jax
import numpy as np
import pytest

X64 = os.environ.get("PROPAX_TEST_X64", "1") not in ("0", "false", "False")
jax.config.update("jax_enable_x64", X64)

from propax import Interface  # noqa: E402
from propax.fluids._registry import EQS_REGISTRY  # noqa: E402

from . import grids  # noqa: E402
from .tolerances import for_precision  # noqa: E402

SAMPLE = 5
SEED = 42


def _sample_fluids() -> list:
    shipped = sorted(EQS_REGISTRY)
    if not shipped:
        return []
    n = min(SAMPLE, len(shipped))
    drawn = np.random.default_rng(SEED).choice(shipped, size=n, replace=False)
    return sorted(str(name) for name in drawn)


SAMPLED = _sample_fluids()
TRANSCRIPTION, RIGHT_ROOT, RESIDUAL = for_precision(X64)


def coolprop_name(propax_name: str) -> str:
    from CoolProp.CoolProp import get_global_param_string

    canonical = {
        n.lower().replace(" ", ""): n
        for n in get_global_param_string("fluids_list").split(",")
    }
    return canonical[propax_name.lower().replace(" ", "")]


@pytest.fixture(scope="session")
def x64() -> bool:
    return X64


@pytest.fixture(scope="session", params=SAMPLED)
def fluid_name(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def props(fluid_name):
    return Interface.create(fluid_name)


@pytest.fixture(scope="session")
def eos(props):
    return props.eos


@pytest.fixture(scope="session")
def saturation(props):
    return props.saturation


@pytest.fixture(scope="session")
def grid(props):
    """The fluid's domain sampled once, shared by every test that needs states."""
    return grids.build(props)


@pytest.fixture(scope="session")
def dome(saturation):
    """Temperatures along the saturation curve, for the tests that walk it."""
    return grids.dome_line(saturation)


@pytest.fixture(scope="session")
def cp_fluid(fluid_name) -> str:
    return coolprop_name(fluid_name)


def pytest_report_header(config):
    return f"propax: float{'64' if X64 else '32'}, {len(SAMPLED)} fluid(s) sampled"
