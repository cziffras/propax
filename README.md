# propax

[![PyPI](https://img.shields.io/pypi/v/propax.svg)](https://pypi.org/project/propax/)
[![CI](https://github.com/cziffras/propax/actions/workflows/ci.yml/badge.svg)](https://github.com/cziffras/propax/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/cziffras/propax/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://github.com/cziffras/propax/blob/main/pyproject.toml)

<p align="center">
  <img src="https://raw.githubusercontent.com/cziffras/propax/main/imgs/cp_propax.png" alt="illustration" width="400">
</p>

`propax` is a CoolProp-like fluid properties library written in pure JAX. It is
differentiable, jittable and vmappable, runs on CPU and GPU in both 32 and 64 bit
precision, and never calls back to Python through `pure_callback`.

It also aims to be transparent. EOS coefficients are taken from the literature, so
results can be reproduced and compared. The other constants (critical point,
saturation curve) are recomputed offline from those coefficients, so every value
matches the EOS exactly.

## Install

```
pip install propax
```

The runtime is JAX and nothing else of substance. The 126 fluids that convert
today ship with the package.

One extra is for development only, and the runtime never imports it:

```
pip install "propax[coolprop]"   # CoolProp + mpmath, to transcribe new fluids
```

## Usage

```python
import jax
jax.config.update("jax_enable_x64", True)  # True or False, depending on your needs
from propax import Interface

props = Interface.create("n-propane")

# accurate: a bracketed solve with automatic phase detection, no table needed
state, converged = props.flash("P", 2e5, "H", 3e5)

# fast: bicubic table lookup, once the tables are built
state, reliable = props.fast_flash("D", 2.0, "U", 4e5)

# direct EOS evaluation (single phase)
state = props.props_rhoT(2.0, 300.0)

# and it differentiates
dT_dP = jax.grad(lambda p: props.flash("P", p, "H", 3e5)[0]["T"])(2e5)
```

Plain floats work without triggering a recompile, so `propax` can be used like
CoolProp by people who don't know JAX. It composes with `jax.jit`, `jax.vmap` and
`jax.grad` as usual.

Two conventions to know:

- `flash` returns `(state, converged)`. Check the flag.
- A two-phase state has `cv` and `cp` set to NaN. They are undefined there
  (temperature stays fixed while heat is added), and a lever-rule number would
  look like an answer when it is not one.

## How does it relate to CoolProp?

CoolProp is used at development time only: to transcribe fluids, and as the oracle
in the tests. The runtime never imports it.

Other projects, such as [jaxprop](https://github.com/turbo-sim/jaxprop), wrap
CoolProp and expose it through `jax.pure_callback()`. This makes CoolProp callable
from JAX code, but has two costs:

- the calls still run on CPU, which rules out GPU or TPU execution;
- every call pays for array conversion and host/device transfers, and XLA cannot
  optimize across it.

Interpolating CoolProp tables in JAX would avoid the callback, but costs a lot of
memory and loses precision. Derivatives suffer most: autodiff through an
interpolant returns smoothed values, while some true derivatives are already very
small. `propax` therefore solves the EOS directly, with iterative solvers.

Only the EOS coefficients are transcribed, index by index from the published
correlation. Everything else is solved from them:

- **Critical point:** not the published value, but the point where the correlation
  itself has `dP/drho = d2P/drho2 = 0`.
- **Saturation curve:** a Chebyshev superancillary fitted offline in extended
  precision (mpmath) against equal pressure and equal fugacity, not against
  CoolProp.

The offline machinery lives in `src/propax/utils/exact/`.

## Fluid catalogue

126 of CoolProp's 136 pure fluids convert today. The saturation fit takes minutes
per fluid, so the catalogue is generated once on CI by
`.github/workflows/catalogue.yml` and committed, rather than rebuilt by every user.

Every converted fluid gives density, energies, `cp` and `cv`. 16 of them are also
usable end to end, with viscosity and conductivity.

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

Transport is the less complete part. These families are not supported yet, as
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

### Making a fluid

```
pip install "propax[coolprop]"
python -m propax.make_fluid Argon            # transcribe the EOS, then fit its saturation curve
python -m propax.utils.make_utils.coverage   # what converts, and what blocks the rest
```

This is useful to check or change a shipped fluid, or to add one CoolProp gains
after a release.

The converter copies the Helmholtz coefficients (machine-precision EOS), checks the
result against CoolProp to 1e-9, and checks each transport block to 1e-6. A
correlation it cannot reproduce from published coefficients is omitted, never
approximated. With `--transport-scaffold`, the property is kept instead: each slot
CoolProp hardcodes gets a placeholder and its literature reference, ready to be
implemented.

## Interpolation tables

`Interface.flash` uses bracketed iterative solvers: robust and precise, but slower
than a lookup (still about 2.5 times faster than `CoolProp.PropsSI`, on CPU). `fast_flash`
uses bicubic tables instead, built with the package's own solvers, with no CoolProp
dependency. Their derivatives come from autodiff, which also makes the bicubics
more precise.

```
python -m propax.build_tables n-propane --pair P H --pair D U \
    --bounds P=1e4:5e6 H=-3e5:1.5e6 D=3.5:717 U=-2.3e5:1.4e6
python -m propax.build_tables --list                   # what the cache holds, and its size
python -m propax.build_tables --remove argon           # free one fluid
python -m propax.build_tables --clear                  # free everything
```

- **Coverage:** only the pairs you ask for, over the bounds you give for each of
  their variables. Outside them `fast_flash` returns NaN; inside, next to
  states the solvers cannot reach, its values are extrapolated. Its second
  output is False in both cases.
- **Size:** a few hundred MB per fluid, which is why tables are built on demand
  rather than shipped.
- **Location:** `~/.cache/propax/<fluid>/`, or anywhere with `PROPAX_TABLE_DIR` or
  `--table-dir`.
- **Precision:** `--target` (1e-5 by default) adapts the grids to the lightest ones
  meeting that tolerance. Tables are designed for float32-level tolerances
  (~1e-5 to ~1e-7 rtol). When you need more, use `flash`, which needs no table.

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

## Roadmap

### Extended corresponding states

The coefficients are published and the reference fluids are already exact here.
ECS is what separates 16 fluids usable end to end from most of the refrigerant
catalogue, so it comes first.

### Mixtures

The next big extension is the standard multi-fluid Helmholtz mixture model of
[GERG-2008 (Kunz & Wagner 2012)](https://doi.org/10.1021/je300655b), the
formulation used by REFPROP and
[CoolProp's mixture backend](https://coolprop.org/fluid_properties/Mixtures.html#theoretical-description).
It builds on the existing pure-fluid equations, adding reducing functions and
binary departure terms. The first version will support the (T, P) flash only.

## License

MIT.
