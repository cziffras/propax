# Reference

JAX must be in 64-bit mode before anything else, or `create` raises:

```python
import jax
jax.config.update("jax_enable_x64", True)

from propax import Interface
props = Interface.create("n-propane")
```

## The interface

::: propax.Interface
    options:
      members:
        - create
        - flash
        - fast_flash
        - props_rhoT
        - check_phase
        - supported_pairs

### What comes back

`flash`, `fast_flash` and `props_rhoT` all return a `ResultDict`,
indexed by the CoolProp-style name  `state["T"]`, `state["rho"]`.

| key | quantity | unit |
|---|---|---|
| `P` | pressure | Pa |
| `T` | temperature | K |
| `rho` | mass density | kg/m³ |
| `h` `u` | enthalpy, internal energy | J/kg |
| `s` | entropy | J/kg/K |
| `cv` `cp` | heat capacities | J/kg/K |
| `viscosity` | dynamic viscosity, `transport=True` only | Pa·s |
| `conductivity` | thermal conductivity, `transport=True` only | W/m/K |

### Which pairs work

Eleven of the twenty-one input pairs, each answered by one of four routes.
`supported_pairs()` returns exactly this mapping, read off the same tables the
dispatch branches on:

| route | pairs | how |
|---|---|---|
| `natural` | `(D,T)` | no solve: it is the state already |
| `saturated` | `(P,Q)` `(T,Q)` | read off the saturation curve, lever rule |
| `one_dim` | `(D,U)` `(D,H)` `(D,S)` `(T,P)` `(T,S)` | one bracketed solve for whichever of `D`/`T` is missing |
| `nested` | `(P,H)` `(P,S)` `(P,U)` | a bracket on `T` enclosing one on density |

The other ten are refused with their reason, in `flash.dispatch.INVALID_PAIRS`:
`(H,S)` and `(S,U)` are two caloric variables with an ill-conditioned Jacobian,
`(T,H)` and `(T,U)` are not injective in density, `(P,D)` has no bracket in
either ordering. [flash.md](flash.md) proves the monotonicity each bracket
relies on.

## The saturation curve

`props.saturation` is a `Superancillary`: the Chebyshev fit of the coexistence
curve, solved offline from the EOS rather than fitted to anyone's output. It
carries `T_min` (the triple point), `T_crit` (solved, not the published value)
and `rho_max`.

::: propax.core.saturation.Superancillary
    options:
      members:
        - state_T
        - state_P
        - densities
        - T_of_rho
        - inside_dome
        - s_of_T
        - T_of_s

A `SaturationResult` carries `P`, `T`, `is_valid`, and the two branches `L` and
`V`, each a dict of `rho h s u`. `is_valid` covers `[T_min, T_crit]` closed at
both ends and is false outside it below the triple point and above the
critical temperature there is no equilibrium to report, so remind to always read the flag before
the branches.

## Custom transport terms

For a correlation propax has no class for  one CoolProp hardcodes, say 
register a JAX module and name it from the fluid file. The complete loop is
`tests/test_custom_transport.py`.

```python
import equinox as eqx
from propax import register_viscosity_higher_order

class MuznyH2(eqx.Module):
    a: float

    def contribution(self, rho_molar, T, eta0):   # -> Pa.s
        return self.a * rho_molar * eta0

register_viscosity_higher_order("hydrogen_higher_order", MuznyH2)
```

The slot's `params` are the constructor's keyword arguments, so the coefficients
live in the file and the equation lives in Python. Register a function instead
of a class and it is called as `builder(params, *, eos)`; or
`(params, *, eos, viscosity)` for a conductivity residual; so a term needing the
EOS can reach it. Loading a fluid whose file names an unregistered term raises
with the name and the exact signature it owes.

::: propax.register_viscosity_higher_order

::: propax.register_conductivity_residual

## Command line

```
python -m propax.make_fluid <Name>                    transcribe a fluid, fit its curve
python -m propax.make_fluid --list-convertible        the names this converter accepts
python -m propax.build_tables <fluid> --target 1e-5   build its interpolation tables
python -m propax.build_tables --list                  what the cache holds, and its size
python -m propax.utils.make_utils.coverage            what converts, and what blocks the rest
```

`make_fluid` needs the `coolprop` extra; the runtime never imports it. Pass
`--target` when building tables: without it the grid is uniform and the
two-phase tables are skipped, so `fast_flash` returns NaN inside the dome.

## Architecture

`propax` is written to be inspected, to be clear to anyone that embeds it in a physical model, below is the library's architecture.

```
propax/
  core/
    interface.py      Interface, the facade and the only public class
    config.py         ThermoVar, PhaseID, the table registry
    saturation.py     Superancillary: the fitted curve at runtime
    interp.py         BicubicInterpolation, ChebyshevPieces
    domain.py         where the correlation is valid
    tolerances.py     every tolerance and iteration cap, in one place
    flash/
      dispatch.py     which route a pair takes, and which pairs are refused
      single_phase.py the bracketed 1D solve and the nested one
      two_phase.py    the lever rule, and the three ways T_sat is found
      results.py      a solved state into the returned dict
      tables.py       the optional interpolation accelerator
  fluids/
    schema.py         the pydantic schema for a definition file
    _registry.py      every file in data/ becomes a set of factories
    data/*.json       one file per fluid: coefficients and the fitted curve
    generic/          one module per correlation family
  utils/
    numerics.py      
    solvers.py        the bracketed Newton and bisection
    exact/            offline, mpmath: critical point and superancillary fit
    make_utils/       offline: CoolProp -> a definition file
    build_utils/      offline: a definition file -> interpolation tables
```

The three `utils` subpackages are dev-time only and never imported by the
runtime: `exact` needs mpmath, `make_utils` needs CoolProp, `build_utils` needs
SciPy.

