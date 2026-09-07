# propax

Reference-quality fluid properties in pure JAX : ready to be embedded in any ML workflow ! 

```python
import jax
jax.config.update("jax_enable_x64", True)   # required, before anything else

from propax import Interface

props = Interface.create("n-propane")
state, converged = props.flash("P", 2e5, "H", 3e5)
dT_dP = jax.grad(lambda p: props.flash("P", p, "H", 3e5)[0]["T"])(2e5)
```

As stated before, every property is evaluated from the fluid's own multiparameter Helmholtz equation of state. The coefficients are transcribed from the published correlation directly from CoolProp's library. Two quantities are then derived instead of trusting the limits provided by CoolProp : critical point properties and saturation superancillary, both are solved off-line in extended precision.

Every flash is itself bracketed and returns a convergence flag without returning a NaN that could kill long computation, gradients are evaluated through the implicit function theorem to save memory.

## Where to go

- **[Reference](reference.md)** is an API overview.
- **[Equations of state](eos.md)** presentation of the Helmholtz formulation and each term family.
- **[The flash](flash.md)** how flashes themselves are currently solved every pair.
- **[Superancillary](superancillary.md)** the Chebyshev machinery and the
  critical expansion.
- **[Transport](transport.md)** viscosity and conductivity, slot by slot comparable to eos.md but for transport properties.

Also take a look at `tutorials/explore.ipynb`, which re-runs lightly on your own machine.
