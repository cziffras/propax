# **PROPAX**

[![CI](https://github.com/cziffras/propax/actions/workflows/ci.yml/badge.svg)](https://github.com/cziffras/propax/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/cziffras/propax/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://github.com/cziffras/propax/blob/main/pyproject.toml)

<p align="center">
  <img src="https://raw.githubusercontent.com/cziffras/propax/main/imgs/cp_propax.png" alt="illustration" width="400">
</p>



Many thermodynamics simulations could make use of an easy-to-use CoolProp-like interface that
is fully differentiable and compatible with the greater JAX ecosystem. It is exactly what `propax` is : written in pure JAX, fully differentiable, jittable and vmappable thermodynamics without any `pure_callback` running on CPU/GPU in both 32 and 64 bits precision ! 

Additionally `propax` aims at being transparent and theoretically solid, EOS parameters are taken from litterature for reproducibility and comparability but
most physical constants (saturation properties, critical point...) are recomputed
offline so that all values do perfectly match the EOS structure and nothing remains hidden.

## To what extent does this work rely on CoolProp ? 

Other similar projects (see [jaxprop](https://github.com/turbo-sim/jaxprop)) are essentially wrapping CoolProp itself and expose it via `jax.pure_callback()`. While making CoolProp calls possible in JAX-written code, this causes two distinct issues : 
- CoolProp calls are still performed on CPU which blocks any different hardware usage (GPU, TPU...)
- Calls made on CPU causes different performance costs : numpy array conversion, latency due to data movements between CPU Host memory and GPU memory and suboptimality of XLA optimization of your code among others...

This is the reason why I made the choice to write `propax` as an alternative to wrappers without sacrificing the precision reached by iterative solvers (one could have implemented a simple interpolation of CoolProp tables in JAX, however the cost in memory is heavy and precision is dubious mostly for what it comes to derivatives computed via autodiff which are smoothed by any interpolation while they can sometimes be extremely small already).

In `propax`, the EOS coefficients are transcribed index by index from the
published correlation, and everything else is solved from them :

- The critical point : is not the published value, it is the point
  where the correlation itself has `dP/drho = d2P/drho2 = 0`. 
- The saturation curve : is a Chebyshev superancillary, fitted offline in
  extended precision (mpmath) against equal pressure and equal fugacity, not
  against CoolProp.

The offline machinery lives in `utils/exact/` and is documented in
`docs/superancillary.md`.


## Making a fluid

```
pip install "propax[coolprop]"               # CoolProp + mpmath, dev-time only
python -m propax.make_fluid Argon            # transcribe the EOS, then fit its saturation curve
python -m propax.utils.make_utils.coverage   # what is convertible, and what blocks the rest
```

The converter copies the Helmholtz coefficients (machine-precision
EOS), checks the result back against CoolProp to 1e-9, and checks each transport
block to 1e-6. Please note that a correlation it cannot reproduce from published coefficients
is omitted, never approximated. In that case, the option `--transport-scaffold`
keeps the property instead of dropping it leaving each slot hardcoded in `CoolProp` with a placeholder and its reference in the litterature so you can implement it.

126 of CoolProp's 136 pure fluids convert today, and the package ships the ones that do, so
`pip install propax` is all you need for them. The saturation fit costs minutes per fluid, so the
catalogue is generated once on CI by `.github/workflows/catalogue.yml` and committed rather than
rebuilt by everyone. `make_fluid` above stays useful for checking one, for changing one, and for
a fluid CoolProp gains after a release. Ten fluids are blocked, five by a residual term family
and five by an ideal-gas one, that are to be implemented soon :

| Blocked | Fluids | Missing family |
|---|---|---|
| residual | water, CO₂ | `ResidualHelmholtzNonAnalytic` (Span–Wagner critical terms) |
| residual | ammonia | `ResidualHelmholtzGaoB` |
| residual | methanol | `ResidualHelmholtzDoubleExponential` |
| residual | R125 | `ResidualHelmholtzLemmon2005` |
| ideal | D6, n-heptane | `IdealGasHelmholtzCP0AlyLee` |
| ideal | air, fluorine | `IdealGasHelmholtzPlanckEinsteinGeneralized` |
| ideal | n-undecane | `CP0PolyT` with `t = -1` (integrates to `tau*ln(tau)`) |

Transport is the sparser half. Families **not yet supported**,
verified against the `model`/`type` tags in
[CoolProp's fluid definition files](https://github.com/CoolProp/CoolProp/tree/master/dev/fluids):

| Part         | Family                                                                 | Blocks e.g.                          | Source |
|--------------|------------------------------------------------------------------------|--------------------------------------|--------|
| Visc./cond.  | extended corresponding states (ECS); CoolProp's fallback for fluids with no dedicated correlation | siloxanes, most refrigerants | [Huber, Laesecke & Perkins 2003](https://doi.org/10.1021/ie0300880) |
| Viscosity    | friction theory higher-order term                                       | several alkanes                      | [Quiñones-Cisneros et al. 2000](https://doi.org/10.1016/S0378-3812(00)00474-X) |
| Viscosity    | Chung et al. corresponding-states                                       | minor fluids                         | [Chung et al. 1988](https://doi.org/10.1021/ie00076a024) |
| Visc./cond.  | hardcoded per-fluid schemes (IAPWS water, Laesecke CO₂, hard-sphere alkanes) | water, CO₂ (viscosity), n-hexane, n-heptane | [Huber et al. 2009](https://doi.org/10.1063/1.3088050); [Laesecke & Muzny 2017](https://doi.org/10.1063/1.4977429); [Michailidou et al. 2013](https://doi.org/10.1063/1.4818980) |
| EOS          | cubic (SRK, Peng–Robinson) and PC-SAFT                                 | quick approximate fluids, mixtures   | [Soave 1972](https://doi.org/10.1016/0009-2509(72)80096-4); [Peng & Robinson 1976](https://doi.org/10.1021/i160057a011); [Gross & Sadowski 2001](https://doi.org/10.1021/ie0003887) |

More will be supported soon too (ECS is planned first). Today 16 fluids are usable end to end (EOS **and** viscosity
**and** conductivity); every one of the 126 gives density, energies and cp/cv.

## Interpolation tables

The runtime solvers (`Interface.flash`) are bracketed and iterative,
so they are robust and precise but somewhat slow (still 2.5 times the speed of `CoolProp.PropsSI`). 
To tackle this `propax`implements bicubic interpolation with tunable precision (tunable floating point
precision, tunable tolerance, absolutely no `CoolProp` dependency, derivatives
via autodiff which also allows for bicubics to be a lot more precise) :

```
python -m propax.build_tables hydrogen --target 1e-5   # only the fluids you use
python -m propax.build_tables --list                   # what the cache holds, and its size
python -m propax.build_tables --remove argon           # free one fluid
python -m propax.build_tables --clear                  # free everything
```

A fluid's tables cost a few hundred MB, which is why these, unlike the fluid
definitions, are built on demand rather than shipped: name the ones you need.
Tables land in `~/.cache/propax/<fluid>/` and `PROPAX_TABLE_DIR` (or
`--table-dir`) moves the cache anywhere you want.

Tables are built natively with the package's own solvers, and `--target` 
adaptively shrinks / widens the computation grids to match a tolerance (with 1e-5 by default) while
being the lightest possible. A table is not meant to meet high precision
requirements anyway, by design it suits float32-compatible tolerances (~1e-5 to
~1e-7 rtol); when you need more, `flash` needs no table at all.

## More on performance

What makes JAX both great to use and difficult to optimize is its vectorization mechanism. `jax.vmap(func)` returns a vectorized version of a scalar function, this must however not be confused with C++ vectorization. JAX performs SIMD/SIMT operations, when solving 100K flashs at once the slowest solve pins down all of the others, slowing down a lot the computations, while C++ vectorization (or standard CPU multi-threading) processes each element independently. This must be kept in mind if you are seeking performance increase. Performance measurements are plotted in
[tutorials/explore.ipynb](https://github.com/cziffras/propax/blob/main/tutorials/explore.ipynb), which also walks through the
phase envelope, a `c_p` field and the Joule–Thomson inversion curve. 

## Usage

```python
import jax
jax.config.update("jax_enable_x64", True)  # required, before anything else
from propax import Interface
props = Interface.create("n-propane")

# accurate: bracketed solve, no table needed (works across the biphasic dome by
# doing automatic phase detection and runs at various speeds across flash types)
state, converged = props.flash("P", 2e5, "H", 3e5)

# fast: bicubic table lookup only, once you have built the tables
state = props.fast_flash("D", 2.0, "U", 4e5)

# direct EOS evaluation (single phase)
state = props.props_rhoT(2.0, 300.0)

# and it differentiates, of course
dT_dP = jax.grad(lambda p: props.flash("P", p, "H", 3e5)[0]["T"])(2e5)
```

Note that `propax` can be used by simply passing floats (traditionally considered as static inputs in JAX) without triggering `jit` recompile allowing `CoolProp`-like usage for users that are not used to JAX. Obviously it itself naturally composes with `jax.jit`, `jax.vmap` and `jax.grad`.

Two conventions worth knowing : `flash` returns `(state, converged)` and you
are meant to look at the flag, and a two-phase state comes back with `cv` and `cp`
set to NaN, because they are genuinely undefined there (temperature is pinned
while heat is added) and I would rather say so than return a lever-rule number
that looks like an answer.

## Install & discover

```
pip install propax
```

That is the whole runtime: JAX and nothing else of substance. The extras are
dev-time only, and neither is imported by the runtime :

```
pip install "propax[coolprop]"   # CoolProp + mpmath, to transcribe new fluids
pip install "propax[build]"      # scipy, to build interpolation tables
```

From source instead, to run the tests or serve the docs :

```
git clone https://github.com/cziffras/propax && cd propax
uv sync --extra test
uv run pytest
```

Documentation renders from the docstrings, so `uv run mkdocs serve` gives you
the browsable version of `docs/reference.md`.

The theory, from the first principles up, is in `docs/` :
`eos.md` for the Helmholtz formulation,
`flash.md` for the flashes and why each bracket is valid,
`superancillary.md` for the Chebyshev machinery and the
critical expansion, `transport.md` for the viscosity and
conductivity families. 

## Tests

```
python -m pytest tests/ -v
```

The CoolProp-based tests are skipped unless CoolProp is installed
(`uv sync --extra coolprop`); CoolProp is only ever used as a test oracle and
is never imported by the runtime.

## Roadmap: what's coming next ?

### Extended corresponding states

Described above : the coefficients are published, the reference fluids are already
exact here, and it is the difference between 16 fluids usable end to end and most
of the refrigerant catalogue.

### Mixtures

The next big extension is the standard multi-fluid Helmholtz mixture
model of [GERG-2008 (Kunz & Wagner 2012)](https://doi.org/10.1021/je300655b),
the formulation used by REFPROP and
[CoolProp's mixture backend](http://www.coolprop.org/fluid_properties/Mixtures.html) ([relevant documentation](https://coolprop.org/fluid_properties/Mixtures.html#theoretical-description)):

## License

MIT.
