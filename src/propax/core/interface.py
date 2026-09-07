import logging
from typing import Dict, Optional, Tuple, Type

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from propax.fluids._registry import EQS_REGISTRY
from propax.fluids.generic import HelmholtzEOS
from propax.utils.types import jaxBool, jaxFloat, jaxInt

from ..utils.numerics import safe_div
from ._state import PropertyMap
from .config import (
    TABLE_REGISTRY,
    PhaseID,
    ThermoVar,
    get_table_path,
)
from .flash import (
    add_transport,
    both_natural_pair,
    call_interp,
    check_supported,
    fill_result_dict,
    is_degenerate_pair,
    is_nested_pair,
    is_saturated_pair,
    one_dim_known_var,
    properties_Tx,
    supported_pairs,
    undefine_two_phase,
)
from .flash.single_phase import solve_1d, solve_nested
from .flash.two_phase import solve_saturated, solve_two_phase
from .interp import BicubicInterpolation
from .saturation import Superancillary

logger = logging.getLogger(__name__)


class Interface(eqx.Module):
    """Every property of one fluid, from any supported pair of inputs.

    Two routes reach the same answer. `fast_flash` reads a prebuilt table and
    is the one to reach for inside an integrator, where the call count is what
    costs; `flash` brackets and solves from the EOS, needs nothing built, and
    is what the tables themselves are checked against.
    """

    eos: HelmholtzEOS
    # None when the fluid has no transport correlation, or when the interface
    # was built with with_transport=False (EOS-only)
    viscosity: Optional[eqx.Module]
    conductivity: Optional[eqx.Module]
    saturation: Superancillary

    # Stores all loaded vectorized tables: "Table_D_U" -> BicubicInterpolation
    interpolators: Dict[str, BicubicInterpolation]

    # which table answers a given unordered pair
    table_for_pair: Dict[frozenset, str] = eqx.field(static=True)

    # what `build_tables` would be given to fill in a missing table
    fluid_name: str = eqx.field(static=True, default="")

    @classmethod
    def supported_pairs(cls) -> Dict[Tuple[str, str], str]:
        """
        The routes are `saturated` (a quality against P or T, read straight off
        the curve), `natural` ((D, T), which needs no solve at all), `one_dim`
        (a bracketed solve for whichever of D or T is missing) and `nested` (a
        bracket on T enclosing one on density).
        """
        return {
            tuple(sorted(v.value for v in pair)): route  # type: ignore[misc]
            for pair, route in supported_pairs().items()
        }

    @classmethod
    def create(
        cls,
        fluid_name: str,
        table_dtype: Type[jnp.dtype] = jnp.float32,
        with_transport: bool = True,
    ) -> "Interface":
        """Build the interface for one fluid, loading whatever tables exist.

        Args:
            fluid_name: A registered fluid. Case and spaces are ignored.
            table_dtype: Storage precision of the interpolation tables.
                float32 by default, which halves device memory; the tables
                only seed the solvers, so final accuracy comes from them and
                not from this. Pass jnp.float64 to store at full precision.
            with_transport: Build the viscosity and conductivity modules.
                They are absent anyway when the fluid's definition file
                carries no such correlation, which is the common case. Either
                way every EOS property still works; only
                `flash(..., transport=True)` needs them.

        Raises:
            ValueError: if no fluid of that name is registered.
        """
        if not jax.config.read("jax_enable_x64"):
            logger.warning(
                "JAX is in 32-bit precision, so every solve stops at float32 "
                "round-off (~1e-7 relative) against ~1e-15 in float64, and "
                "`TOL32` is used rather than `TOL64`. Enable 64-bit "
                "unless you meant this:\n"
                "import jax\n"
                "jax.config.update('jax_enable_x64', True)"
            )
        key_name = fluid_name.lower().replace(" ", "")

        if key_name not in EQS_REGISTRY:
            raise ValueError(
                f"no fluid named {fluid_name!r} is registered; propax ships "
                + ", ".join(sorted(EQS_REGISTRY))
            )

        eos_cls, _, viscosity_cls, conductivity_cls = EQS_REGISTRY[key_name]
        eos = eos_cls()
        saturation = Superancillary.create(eos, key_name)

        viscosity = viscosity_cls(eos=eos) if with_transport else None
        conductivity = (
            conductivity_cls(eos=eos, viscosity=viscosity)
            if viscosity is not None
            else None
        )

        interpolators = {}
        table_for_pair = {}
        missing: list = []

        built = get_table_path() / key_name

        # Every table is loaded: `flash` brackets its way to the answer and
        # asks for none of them, so the only caller left is `fast_flash`,
        # which wants whichever pair it is handed. The dome table beside each
        # one carries the mixture branch on its own axes, without which a
        # lookup inside the dome reads the single-phase surface and is wrong.
        def load(name: str) -> BicubicInterpolation:
            return BicubicInterpolation.create(str(built / name), dtype=table_dtype)

        for spec in TABLE_REGISTRY:
            dome = f"{spec.name}_dome"
            try:
                interpolators[spec.name] = load(spec.name)
                interpolators[dome] = load(dome)
            except (OSError, ValueError) as why:
                missing.append(f"{spec.name} ({why})")
                continue
            axes = (spec.x_axis.variable, spec.y_axis.variable)
            table_for_pair[frozenset(axes)] = spec.name

        # Not an error: `flash` brackets its way to the answer without
        # reading a table, so a fresh install is usable before anything is
        # built. Only `fast_flash` needs them, and it says so itself.
        if missing:
            logger.info(
                "%s: %d of %d table(s) not loaded from %s, `fast_flash` is "
                "unavailable for them until `python -m propax.build_tables "
                "%s` has run. Missing: %s",
                key_name,
                len(missing),
                len(TABLE_REGISTRY),
                built,
                key_name,
                "; ".join(missing),
            )

        return cls(
            eos=eos,
            viscosity=viscosity,
            conductivity=conductivity,
            saturation=saturation,
            interpolators=interpolators,
            table_for_pair=table_for_pair,
            fluid_name=key_name,
        )

    @eqx.filter_jit
    def _interp(self, tvar1, val1, tvar2, val2) -> PropertyMap:
        """The interface's one door onto the interpolation tables."""
        return call_interp(
            self.interpolators, self.table_for_pair, tvar1, val1, tvar2, val2
        )

    @eqx.filter_jit
    def props_rhoT(self, rho_mass: Array, T: Array) -> PropertyMap:
        """Every property at the EOS's own variables, with no solve at all.

        Args:
            rho_mass: Mass density, kg/m3.
            T: Temperature, K.

        Returns:
            The properties. No convergence flag comes with them, because
            there is nothing to converge; outside the domain every field is
            NaN.
        """
        return self.eos.props_rhoT(rho_mass, T)

    def check_phase(
        self,
        name1: str,
        val1: jaxFloat,
        name2: str,
        val2: jaxFloat,
    ) -> jaxInt:
        """Name the phase at a state without returning its properties.

        Args:
            name1: What the first value measures, as a CoolProp-style name
                or the `ThermoVar` itself.
            val1: Its value, in SI.
            name2: The second variable, named the same way.
            val2: Its value, in SI.

        Returns:
            The matching `PhaseID`, and `UNKNOWN` where the state could not
            be placed at all. Plain floats are cast on the way in, so this
            costs no retrace.
        """

        if jax.config.read("jax_enable_x64"):
            dtype = jnp.float64
        else:
            dtype = jnp.float32

        tvar1 = ThermoVar(name1)
        tvar2 = ThermoVar(name2)
        x1 = jnp.asarray(val1, dtype=dtype)
        x2 = jnp.asarray(val2, dtype=dtype)
        return self._phase_of(
            tvar1=tvar1,
            val1=x1,
            tvar2=tvar2,
            val2=x2,
        )

    @eqx.filter_jit
    def _phase_of(
        self,
        tvar1: ThermoVar,
        val1: Array,
        tvar2: ThermoVar,
        val2: Array,
    ):
        is_biphasic, _, _ = solve_two_phase(
            self.eos,
            self.saturation,
            tvar1=tvar1,
            val1=val1,
            tvar2=tvar2,
            val2=val2,
            is_inactive=False,
        )

        def under_the_dome(_):
            return jnp.array(PhaseID.TWO_PHASE)

        def off_the_dome(_):
            properties, _ = self._flash_impl(
                tvar1=tvar1,
                val1=val1,
                tvar2=tvar2,
                val2=val2,
                monophasic=True,
            )
            T, P = properties[ThermoVar.T], properties[ThermoVar.P]
            P_sat = self.saturation.state_T(jnp.minimum(T, self.saturation.T_crit)).P

            # two independent yes/no questions: is T past the critical point,
            # and is the state on the dense side of the line that matters at
            # that T (the critical pressure above P_crit, the saturation
            # pressure below it)
            hot = T >= self.eos.T_crit
            dense = jnp.where(hot, P >= self.eos.P_crit, P >= P_sat)

            named = jnp.array(
                [
                    [PhaseID.GAS, PhaseID.LIQUID],
                    [PhaseID.SUPERCRITICAL_GAS, PhaseID.SUPERCRITICAL],
                ]
            )
            return jnp.where(
                jnp.isfinite(T) & jnp.isfinite(P),
                named[hot.astype(jnp.int32), dense.astype(jnp.int32)],
                jnp.array(PhaseID.UNKNOWN),
            )

        final_phase_id = jax.lax.cond(is_biphasic, under_the_dome, off_the_dome, None)

        return final_phase_id

    def fast_flash(
        self,
        name1: str,
        val1: jaxFloat,
        name2: str,
        val2: jaxFloat,
        transport: bool = False,
    ) -> PropertyMap:
        """One bicubic lookup, then every property rederived from the state it locates.

        Roughly twenty times the throughput of `flash`, at table-grade
        accuracy, and differentiable: the interpolant is differentiated, not
        the data.

        Args:
            name1: What the first value measures, as a CoolProp-style name
                or the `ThermoVar` itself.
            val1: Its value, in SI.
            name2: The second variable, named the same way.
            val2: Its value, in SI.
            transport: Also return viscosity and conductivity.

        Returns:
            The properties alone, with no convergence flag beside them: a
            lookup CANNOT fail loudly.

        Raises:
            FileNotFoundError: if no table has been built for this pair. The
                message names the command that builds it.
        """
        if jax.config.read("jax_enable_x64"):
            dtype = jnp.float64
        else:
            dtype = jnp.float32

        tvar1 = ThermoVar(name1)
        tvar2 = ThermoVar(name2)
        if frozenset({tvar1, tvar2}) not in self.table_for_pair:
            raise FileNotFoundError(
                f"no interpolation table for ({tvar1.value}, {tvar2.value}). Build "
                f"them with `python -m propax.build_tables {self.fluid_name}`, or use "
                "`flash`, which needs none."
            )
        x1 = jnp.asarray(val1, dtype=dtype)
        x2 = jnp.asarray(val2, dtype=dtype)
        state = self._interp(tvar1=tvar1, val1=x1, tvar2=tvar2, val2=x2)
        return self._derive_from_state(state, dtype=dtype, transport=transport)

    def _derive_from_state(self, state, dtype, transport: bool):
        """Full property set from the interpolated (rho, T) --> returns a coherent
        thermo state
        """
        rho = jnp.asarray(state[ThermoVar.D])
        T = jnp.asarray(state[ThermoVar.T])

        # rho against the saturated densities at T names the phase: no quality
        # channel needed, and no dome mask to interpolate across
        sat = self.saturation.state_T(T)
        rho_L, rho_V = sat.L[ThermoVar.D], sat.V[ThermoVar.D]
        two_phase = sat.is_valid & (rho < rho_L) & (rho > rho_V)

        v, v_L, v_V = 1.0 / rho, 1.0 / rho_L, 1.0 / rho_V
        x = jnp.clip(safe_div(v - v_L, v_V - v_L), 0.0, 1.0)

        one = add_transport(
            self.viscosity,
            self.conductivity,
            self.props_rhoT(rho, T),
            transport,
        )
        two = properties_Tx(
            self.saturation,
            jnp.where(two_phase, T, self.eos.T_crit),
            x,
        )
        two_dict = two.mix.replace({ThermoVar.P: two.P, ThermoVar.T: two.T})
        two_dict = add_transport(
            self.viscosity,
            self.conductivity,
            two_dict,
            transport,
        )

        # cv and cp are not defined for a two-phase mixture (T is pinned while
        # heat is added), fill result dict fills their slots with nans for avoidind
        # crashing a simulation a downstream user could route nans to 0.0 or use
        # jnp.nan_to_num : https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nan_to_num.html
        # this is done for clarifying code's intents
        undefined = {ThermoVar.CVMASS.internal_key, ThermoVar.CPMASS.internal_key}
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        merged = {
            k: jnp.where(two_phase, nan if k in undefined else two_dict[k], v)
            if (k in two_dict or k in undefined)
            else v
            for k, v in one.items()
        }
        return PropertyMap(fill_result_dict(merged, dtype=dtype, transport=transport))

    def flash(
        self,
        name1: str,
        val1: jaxFloat,
        name2: str,
        val2: jaxFloat,
        monophasic: bool = False,
        transport: bool = False,
    ) -> Tuple[PropertyMap, jaxBool]:
        """Solve the state from any supported pair, and report every property.

        Whichever pair is handed in, and whether the state lands on the dome
        or off it, the same set of properties comes back. Scalars are cast on
        the way in, so passing Python floats costs no retrace.

        Args:
            name1: What the first value measures, as a CoolProp-style name
                ("P", "T", "D", ...) or the `ThermoVar` itself.
            val1: Its value, in SI.
            name2: The second variable, named the same way.
            val2: Its value, in SI.
            monophasic: Skip the two-phase test and go straight to the
                single-phase solve, for a caller that already knows the state
                is off the dome.
            transport: Also return viscosity and conductivity. The fluid must
                carry a correlation for them, or this raises.

        Returns:
            The properties, keyed by internal name ("P", "T", "rho", "u",
            "h", "s", "cv", "cp", plus "viscosity" and "conductivity" under
            `transport`), and a flag saying whether the solve landed on a
            physical state. Under the dome "cv" and "cp" come back NaN.

        Raises:
            ValueError: if the pair is not one propax can solve, or if
                `transport` is asked of a fluid that has no correlation.
        """
        if jax.config.read("jax_enable_x64"):
            dtype = jnp.float64
        else:
            dtype = jnp.float32

        tvar1 = ThermoVar(name1)
        tvar2 = ThermoVar(name2)

        x1 = jnp.asarray(val1, dtype=dtype)
        x2 = jnp.asarray(val2, dtype=dtype)  # jit and vmap here
        return self._flash_impl(
            tvar1=tvar1,
            val1=x1,
            tvar2=tvar2,
            val2=x2,
            monophasic=monophasic,
            transport=transport,
            dtype=dtype,
        )

    @eqx.filter_jit
    def _flash_impl(
        self,
        tvar1: ThermoVar,
        val1: Array,
        tvar2: ThermoVar,
        val2: Array,
        monophasic: bool = False,
        transport: bool = False,
        dtype: Type[jnp.dtype] = jnp.float64,
    ) -> Tuple[PropertyMap, jaxBool]:
        """The traced body behind `flash`, past the name and dtype handling."""
        check_supported(tvar1, tvar2)

        if transport and (self.viscosity is None or self.conductivity is None):
            raise ValueError(
                "transport=True requires transport properties, but this interface "
                "has none: it was either built with with_transport=False, or the "
                "fluid definition file carries no viscosity/conductivity "
                "correlation. Use transport=False for EOS-only properties."
            )

        # A quality read against P or T needs no solver at all: the saturation
        # state is the answer and the lever rule places the mixture on it, leaves
        # before any other machinery is built
        if is_saturated_pair(tvar1, tvar2):
            ok, T_sat, x = solve_saturated(
                self.eos,
                self.saturation,
                tvar1,
                val1,
                tvar2,
                val2,
            )
            props = properties_Tx(self.saturation, T_sat, x)
            result = props.mix.replace({ThermoVar.P: props.P, ThermoVar.T: props.T})
            result = add_transport(
                self.viscosity,
                self.conductivity,
                undefine_two_phase(result, dtype),
                transport,
            )
            return (
                PropertyMap(fill_result_dict(result, dtype=dtype, transport=transport)),
                ok,
            )

        # (P, T) admits no two-phase state, and `monophasic` forces the
        # single-phase solver (python static)
        degenerate_pair = monophasic or is_degenerate_pair(tvar1, tvar2)
        if degenerate_pair:
            is_biphasic, T_two_phase, x_two_phase = jnp.array(False), jnp.inf, jnp.inf
        else:
            # One two-phase solve serves as both the phase test and the
            # two-phase branch result
            is_biphasic, T_two_phase, x_two_phase = solve_two_phase(
                self.eos,
                self.saturation,
                tvar1,
                val1,
                tvar2,
                val2,
                is_inactive=jnp.array(False),
            )

        # (D, T) are the EOS' own variables: outside the dome there is nothing
        # to solve, the state is the input. Inside it, `solve_two_phase` case 2
        # already answers from T alone, so this branch only serves the
        # single-phase side.
        # /!\ It converges without an initial guess, so those pairs need
        # no interpolation table at all
        both_natural = both_natural_pair(tvar1, tvar2)

        # P fixes an isobar and the caloric variable is monotone along it, so
        # these are bracketable in two stages rather than needing a seed
        # /!\ It converges without an initial guess, so those pairs need
        # no interpolation table at all
        nested = is_nested_pair(tvar1, tvar2)

        # When D or T is an input, only the other member of (rho, T) is
        # unknown: use the bracketed 1D solve
        # /!\ It converges without an initial guess, so those pairs need
        # no interpolation table at all
        known_var = one_dim_known_var(tvar1, tvar2)

        def _one_phase():
            if both_natural:
                rho = val1 if tvar1 == ThermoVar.D else val2
                T = val1 if tvar1 == ThermoVar.T else val2
                # the bracketed solvers below report `is_inactive` as a failure;
                # this branch runs no solver, so it has to say so on its own
                op_status = (
                    jnp.isfinite(rho)
                    & jnp.isfinite(T)
                    & (rho > 0.0)
                    & jnp.logical_not(is_biphasic)
                )
            elif nested:
                if tvar1 == ThermoVar.P:
                    val_P, tvar_other, val_other = val1, tvar2, val2
                else:
                    val_P, tvar_other, val_other = val2, tvar1, val1
                op_status, rho, T = solve_nested(
                    self.eos,
                    self.saturation,
                    tvar_other,
                    val_P,
                    val_other,
                    is_inactive=is_biphasic,
                )
            else:
                # every remaining pair is bracketed: `check_supported` at the
                # top has already turned away the ones that are not
                if known_var is None:
                    raise ValueError(f"{tvar1}/{tvar2} is not a bracketed pair")
                if tvar1 == known_var:
                    val_known, tvar_other, val_other = val1, tvar2, val2
                else:
                    val_known, tvar_other, val_other = val2, tvar1, val1
                op_status, rho, T = solve_1d(
                    self.eos,
                    self.saturation,
                    known_var,
                    val_known,
                    tvar_other,
                    val_other,
                    is_inactive=is_biphasic,
                )
            one_phase_primary_props = self.props_rhoT(rho, T)

            one_phase_result = add_transport(
                self.viscosity,
                self.conductivity,
                one_phase_primary_props,
                transport,
            )

            filled = fill_result_dict(
                one_phase_result,
                dtype=dtype,
                transport=transport,
            )
            return filled, op_status

        def _two_phase():
            tp_status = is_biphasic
            # Outside the dome the shared solve above does not converge, and its
            # (T, x) are meaningless route them to a physical in-dome point
            T_mid = jnp.asarray((self.eos.T_triple + self.eos.T_crit) / 2.0)
            T_biphase = jnp.where(is_biphasic, T_two_phase, T_mid)
            x_biphase = jnp.where(is_biphasic, x_two_phase, 0.5)
            two_phase_primary_props = properties_Tx(
                self.saturation,
                # we are only interested in the equilibrium properties
                T_biphase,
                x_biphase,
            )  # captures T and x from closure

            two_phase_result = two_phase_primary_props.mix.replace(
                {
                    ThermoVar.P: two_phase_primary_props.P,
                    ThermoVar.T: two_phase_primary_props.T,
                }
            )
            two_phase_result = add_transport(
                self.viscosity,
                self.conductivity,
                undefine_two_phase(two_phase_result, dtype),
                transport,
            )

            filled = fill_result_dict(
                two_phase_result,
                dtype=dtype,
                transport=transport,
            )
            return filled, tp_status

        one_result, one_status = _one_phase()
        # the two-phase answer stands wherever no single-phase root was reached,
        # and every branch reports `is_inactive` as a failure: that is the dome
        # no physical single root can exist (metastable optima)
        is_biphasic = is_biphasic & jnp.logical_not(one_status)
        if degenerate_pair:  # static: the two-phase graph is never built
            return PropertyMap(one_result), one_status

        two_result, two_status = _two_phase()
        final_result = jax.tree.map(
            lambda tp, op: jnp.where(is_biphasic, tp, op), two_result, one_result
        )
        exit_code = jnp.where(is_biphasic, two_status, one_status)

        return PropertyMap(final_result), exit_code
