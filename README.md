# propax

[![PyPI](https://img.shields.io/pypi/v/propax.svg)](https://pypi.org/project/propax/)
[![CI](https://github.com/cziffras/propax/actions/workflows/ci.yml/badge.svg)](https://github.com/cziffras/propax/actions/workflows/ci.yml)
[![CAMPAIGN](https://github.com/cziffras/propax/actions/workflows/campaign.yml/badge.svg)](https://github.com/cziffras/propax/actions/workflows/campaign.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/cziffras/propax/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://github.com/cziffras/propax/blob/main/pyproject.toml)

<p align="center">
  <img src="https://raw.githubusercontent.com/cziffras/propax/main/imgs/cp_propax.png" alt="illustration" width="400">
</p>

`propax` is a CoolProp-like fluid properties library (for pure fluids only) written in pure JAX. It is
differentiable, jittable and vmappable, runs on CPU and GPU in both 32 and 64 bit
precision, and never calls back to Python through `pure_callback`.

## Install

propax is available on PyPI, simply run:

```
pip install propax
```

CoolProp and mpmath are development extras only, also used to transcribe new fluids:

```
pip install "propax[coolprop]"
```

## Usage

```python
import jax
jax.config.update("jax_enable_x64", True)  # True or False, depending on your needs (GPUs are often faster with 32-bit precision)
from propax import Interface

Interface.available_fluids()  # ['1-butene', 'acetone', 'argon', ...]
props = Interface.create("n-propane")

# accurate: a bracketed solve with phase detection (see supported pairs)
state, converged = props.flash("P", 2e5, "H", 3e5)

# fast: bicubic table lookup, once the tables are built
state, reliable = props.fast_flash("D", 2.0, "U", 4e5)

# direct EOS evaluation (single phase)
state = props.props_rhoT(2.0, 300.0)

# and it differentiates
dT_dP = jax.grad(lambda p: props.flash("P", p, "H", 3e5)[0]["T"])(2e5)
```

Plain floats work without triggering a recompile, so `propax` can be used like
CoolProp by people who are not used to JAX. It composes with `jax.jit`, `jax.vmap` and
`jax.grad` as usual for JAX enjoyers!

Two conventions to know:

- `flash` returns `(state, converged)`, so always check the flag.
- A two-phase state has `cv` and `cp` set to NaN. They are undefined there
  (temperature stays fixed while heat is added), and a lever-rule number would
  look like an answer when it is not one.

## How does it relate to CoolProp?

CoolProp is used at development time only: to transcribe fluids, and as a coarse oracle
in the tests (sometimes `CoolProp` and `propax` do not operate at the same precision level which
often favors propax).

Other projects, such as [jaxprop](https://github.com/turbo-sim/jaxprop), wrap
CoolProp and expose it through `jax.pure_callback()`. This makes CoolProp callable
from JAX code, but has mainly three drawbacks:

- the calls still run on CPU, which rules out GPU or TPU execution;
- every call pays for array conversion and host/device transfers;
- the computation graph is split at each conversion, so XLA optimization cannot
  be optimal.

Interpolating CoolProp tables in JAX would avoid the callback, but costs a lot of
memory and loses precision. Derivatives suffer most: autodiff through an
interpolant returns smoothed values, while some true derivatives are already very
small. `propax` therefore solves the EOS directly, with iterative solvers.

Only the EOS coefficients are transcribed, index by index from the published
correlation. Everything else is solved from them:

- **Critical point:** finds the exact point where the correlation
  itself has `dP/drho = d2P/drho2 = 0`.
- **Saturation curve:** a Chebyshev superancillary fitted offline in extended
  precision (mpmath) against equal pressure and equal fugacity, not against
  CoolProp.

The offline machinery can be found in `src/propax/utils/exact/`.

## Fluid catalogue

125 of CoolProp's 136 pure fluids can be converted. The saturation fit takes minutes
per fluid, so the catalogue is generated on CI by `.github/workflows/catalogue.yml`. In practice,
`propax` comes with all its available fluids when installed (same as `CoolProp`).

Every converted fluid gives density, energies, `cp` and `cv`. 16 of them are also
usable end to end, with viscosity and conductivity.

R404A, R407C, R410A, R507A and SES36 are blends that CoolProp describes as
pseudo-pure fluids (`eos.pseudo_pure`). propax treats them as pure fluids, so
their two-phase region is the dome of the pseudo-pure EOS, with no temperature
glide. CoolProp takes their bubble and dew lines from ancillary equations
instead, so the two libraries disagree near and inside the dome.

### Blocked fluids

Ten fluids are blocked by an EOS term family that is not implemented yet:

| Blocked | Fluids | Missing family |
|---|---|---|
| residual | water, CO₂ | `ResidualHelmholtzNonAnalytic` (Span–Wagner critical terms) |
| residual | ammonia | `ResidualHelmholtzGaoB` |
| residual | methanol | `ResidualHelmholtzDoubleExponential` |
| residual | R125 | `ResidualHelmholtzLemmon2005` |
| ideal | D6, n-heptane | `IdealGasHelmholtzCP0AlyLee` |
| ideal | air, fluorine | `IdealGasHelmholtzPlanckEinsteinGeneralized` |
| ideal | n-undecane | `CP0PolyT` with `t = -1` (integrates to `tau*ln(tau)`) |

### Transport coverage

Transport is the least complete part. These families are not supported yet, as
checked against the `model`/`type` tags in
[CoolProp's fluid definition files](https://github.com/CoolProp/CoolProp/tree/master/dev/fluids):

| Part | Family | Blocks e.g. | Source |
|---|---|---|---|
| Visc./cond. | extended corresponding states (ECS), CoolProp's fallback for fluids with no dedicated correlation | siloxanes, most refrigerants | [Huber, Laesecke & Perkins 2003](https://doi.org/10.1021/ie0300880) |
| Viscosity | friction theory higher-order term | several alkanes | [Quiñones-Cisneros et al. 2000](https://doi.org/10.1016/S0378-3812(00)00474-X) |
| Viscosity | Chung et al. corresponding states | minor fluids | [Chung et al. 1988](https://doi.org/10.1021/ie00076a024) |
| Visc./cond. | hardcoded per-fluid schemes (IAPWS water, Laesecke CO₂, hard-sphere alkanes) | water, CO₂ (viscosity), n-hexane, n-heptane | [Huber et al. 2009](https://doi.org/10.1063/1.3088050); [Laesecke & Muzny 2017](https://doi.org/10.1063/1.4977429); [Michailidou et al. 2013](https://doi.org/10.1063/1.4818980) |

Cubic EOS (SRK, Peng–Robinson) and PC-SAFT are not supported either
([Soave 1972](https://doi.org/10.1016/0009-2509(72)80096-4);
[Peng & Robinson 1976](https://doi.org/10.1021/i160057a011);
[Gross & Sadowski 2001](https://doi.org/10.1021/ie0003887)).

## Interpolation tables

`Interface.flash` uses bracketed iterative solvers: robust and precise, but slower
than a lookup (still about 2.5 times faster than `CoolProp.PropsSI`, on CPU). `fast_flash`
uses bicubic tables instead, built with the package's own solvers, with no CoolProp
dependency. Their derivatives come from autodiff, which also makes the bicubics
more precise.

```python
from propax import tables

tables.create_all_tables(
    "n-propane",
    pairs=[("P", "H"), ("D", "U")],
    bounds={"P": (1e4, 5e6), "H": (-3e5, 1.5e6), "D": (3.5, 717), "U": (-2.3e5, 1.4e6)},
    target=1e-5
)
tables.cached_fluids()
tables.clear_cache(["argon"])
tables.clear_cache()
```

The second output of `fast_flash` is False off the tables' axes, where the
properties come back NaN, and next to nodes the build could not solve (near the
critical point, at the edges of the domain), where they are extrapolated.

- **Location:** They are sent to `~/.cache/propax/<fluid>/`, or anywhere with `PROPAX_TABLE_DIR`.
- **Precision:** `target` (1e-5 by default) adapts the grids to the lightest ones meeting that tolerance. Tables are designed for float32-level tolerances. When you need more, use `flash`, which needs no table.

## Performance

`jax.vmap` is not C++ vectorization. JAX runs a batch in lockstep: when solving
100k flashes at once, every element waits for the slowest solve. C++ vectorization
or CPU multithreading handles each element independently. Keep this in mind when
tuning for speed.

Performance measurements are plotted in
[tutorials/explore.ipynb](https://github.com/cziffras/propax/blob/main/tutorials/explore.ipynb),
which also walks through the phase envelope, a `c_p` field and the Joule–Thomson
inversion curve.

## Development

```
git clone https://github.com/cziffras/propax && cd propax
uv sync --extra test
uv run pytest                        # float64
PROPAX_TEST_X64=0 uv run pytest      # float32
```

The CoolProp-based tests are skipped when CoolProp is not installed.

### Making a fluid

```
pip install "propax[coolprop]"
python scripts/make_fluid.py Argon # transcribe the EOS, then fit its saturation curve (check for options with --help)
python scripts/make_fluid.py --list-convertible [e|v|c]
```

This is useful to check or change a shipped fluid, or to add one CoolProp gains
after a release.

A correlation it cannot reproduce from published coefficients can be implemented with `--transport-scaffold`: each slot
CoolProp hardcodes gets a placeholder and its literature reference, ready to be
implemented.

## Roadmap

### Extended corresponding states

The coefficients are published and the reference fluids are already exact here.
ECS is what separates 16 fluids usable end to end from most of the refrigerant
catalogue, so it comes first.

## License

MIT.
