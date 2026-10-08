from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def int_pow(x, k: int):
    """Unroll static int powers to avoid SFU speicific kernels (power, exp...)"""
    if k == 0:
        # the identity of whatever x is: an mpf on the offline path, a JAX
        # array on the runtime one. `jnp.ones_like` would only serve the second
        return x**0
    if k < 0:
        return 1.0 / int_pow(x, -k)
    r, b = None, x
    while k:
        if k & 1:
            r = b if r is None else r * b
        k >>= 1
        if k:
            b = b * b
    return r


MAX_UNROLL = 32


def static_pow(x, e):
    if isinstance(e, int) or (float(e).is_integer() and abs(e) <= MAX_UNROLL):
        return int_pow(x, int(e))
    return x**e


def static_floats(values) -> tuple:
    return tuple(float(v) for v in np.asarray(values, dtype=float).tolist())


def static_exponents(values) -> tuple:
    out = []
    for v in np.asarray(values, dtype=float).tolist():
        out.append(int(round(v)) if float(v).is_integer() else float(v))
    return tuple(out)


def _from_log_derivatives(base, delta, tau, d, t, L_d, dL_d, L_t, dL_t):
    p_d = d / delta + L_d
    p_t = t / tau + L_t
    return (
        base,
        base * p_d,
        base * p_t,
        base * (p_d * p_d - d / (delta * delta) + dL_d),
        base * (p_t * p_t - t / (tau * tau) + dL_t),
        base * p_d * p_t,
    )


def _by_autodiff(f, delta, tau):
    x = jnp.stack([delta, tau])

    def g(z):
        return f(z[0], z[1])

    val, grad = jax.value_and_grad(g)(x)
    hess = jax.jacfwd(jax.grad(g))(x)
    return val, grad[0], grad[1], hess[0, 0], hess[1, 1], hess[0, 1]


class IdealTerm(eqx.Module):
    """One additive contribution to the reduced ideal-gas Helmholtz energy
    alpha0(delta, tau). Subclasses implement `contribution`.

    /!\\ Read eos.md
    """

    def contribution(self, delta, tau):
        raise NotImplementedError

    def tau_derivatives(self, delta, tau) -> Tuple:
        a, _, a_t, _, a_tt, _ = _by_autodiff(self.contribution, delta, tau)
        return a, a_t, a_tt


class ResidualTerm(eqx.Module):
    """One functional family of the residual Helmholtz energy.

    One additive contribution to the reduced residual Helmholtz energy
    alphar(delta, tau). Subclasses implement `contribution`.

    /!\\ Read eos.md
    """

    def contribution(self, delta, tau):
        raise NotImplementedError

    def derivatives(self, delta, tau) -> Tuple:
        return _by_autodiff(self.contribution, delta, tau)


class IdealLead(IdealTerm):
    """ln(delta) + a_log*ln(tau) + a1 + a2*tau

    - Lead + log(tau) + reference offset, all folded into a1/a2/a_log (CoolProp
    IdealGasHelmholtzLead + LogTau + EnthalpyEntropyOffset)

    - a_log is c_v0/R in the low-T limit
    """

    a1: float
    a2: float
    a_log: float

    def contribution(self, delta, tau):
        return jnp.log(delta) + self.a_log * jnp.log(tau) + self.a1 + self.a2 * tau

    def tau_derivatives(self, delta, tau) -> Tuple:
        a = self.contribution(delta, tau)
        return a, self.a_log / tau + self.a2, -self.a_log / (tau * tau)


class IdealPlanckEinstein(IdealTerm):
    """sum n_k * ln(1 - exp(-theta_k * tau))

    - theta_k are reduced Einstein temperatures (CoolProp pe_v / T_c)

    - Covers CoolProp IdealGasHelmholtzPlanckEinstein and ...PlanckEinsteinFunctionT
      (repetition in types)
    """

    n: tuple = eqx.field(static=True)
    theta: tuple = eqx.field(static=True)

    def contribution(self, delta, tau):
        total = jnp.zeros_like(tau)
        for n, theta in zip(self.n, self.theta):
            total = total + n * jnp.log(1.0 - jnp.exp(-theta * tau))
        return total

    def tau_derivatives(self, delta, tau) -> Tuple:
        # d/dtau ln(1 - e^-x) = theta e^-x / (1 - e^-x) with x = theta tau, and
        # one more derivative gives -theta^2 e^-x / (1 - e^-x)^2
        # Be cautious here, prior to this version e^x was used and sometimes led
        # to nan in gradients which led to non converging solves

        a = jnp.zeros_like(tau)
        a_t = jnp.zeros_like(tau)
        a_tt = jnp.zeros_like(tau)
        for n, theta in zip(self.n, self.theta):
            e = jnp.exp(-theta * tau)
            gap = -jnp.expm1(-theta * tau)  # 1 - e^-x
            a = a + n * jnp.log(gap)
            a_t = a_t + n * theta * e / gap
            a_tt = a_tt - n * theta * theta * e / (gap * gap)
        return a, a_t, a_tt


class Polynomial(ResidualTerm):
    """
    sum n * delta^d * tau^t

    - Covers (CoolProp ResidualHelmholtzPower, l = 0)

    The sum is unrolled into a scalar accumulator rather than a `jnp.sum` as done
    in the previous verision: the reduction would otherwise materialise a whole vector
    of terms and split the kernel in two for the sake of summing a handful of numbers !
    """

    n: tuple = eqx.field(static=True)
    d: tuple = eqx.field(static=True)
    t: tuple = eqx.field(static=True)

    def contribution(self, delta, tau, namespace=jnp):
        total = 0.0 * delta * tau
        for n, d, t in zip(self.n, self.d, self.t):
            total = total + n * static_pow(delta, d) * static_pow(tau, t)
        return total

    def derivatives(self, delta, tau) -> Tuple:
        out = [jnp.zeros_like(delta * tau)] * 6
        zero = jnp.zeros_like(delta)
        for n, d, t in zip(self.n, self.d, self.t):
            base = n * static_pow(delta, d) * static_pow(tau, t)
            part = _from_log_derivatives(base, delta, tau, d, t, zero, zero, zero, zero)
            out = [a + b for a, b in zip(out, part)]
        return tuple(out)


class ExponentialDensity(ResidualTerm):
    """sum n * delta^d * tau^t * exp(-delta^p)
    (CoolProp ResidualHelmholtzPower, l = p > 0)."""

    n: tuple = eqx.field(static=True)
    d: tuple = eqx.field(static=True)
    t: tuple = eqx.field(static=True)
    p: tuple = eqx.field(static=True)

    def contribution(self, delta, tau, namespace=jnp):
        groups: dict = {}
        for n, d, t, p in zip(self.n, self.d, self.t, self.p):
            groups.setdefault(p, []).append((n, d, t))

        total = 0.0 * delta * tau
        for p, items in groups.items():
            inner = 0.0 * delta * tau
            for n, d, t in items:
                inner = inner + n * static_pow(delta, d) * static_pow(tau, t)
            total = total + inner * namespace.exp(-static_pow(delta, p))
        return total

    def derivatives(self, delta, tau) -> Tuple:
        out = [jnp.zeros_like(delta * tau)] * 6
        zero = jnp.zeros_like(delta)
        for n, d, t, p in zip(self.n, self.d, self.t, self.p):
            L_d = -p * static_pow(delta, p - 1)
            dL_d = -p * (p - 1) * static_pow(delta, p - 2)
            base = (
                n
                * static_pow(delta, d)
                * static_pow(tau, t)
                * jnp.exp(-static_pow(delta, p))
            )
            part = _from_log_derivatives(base, delta, tau, d, t, L_d, dL_d, zero, zero)
            out = [a + b for a, b in zip(out, part)]
        return tuple(out)


class Gaussian(ResidualTerm):
    """sum n * delta^d * tau^t * exp(phi*(delta-D)^2 + beta*(tau-gamma)^2)

    CoolProp ResidualHelmholtzGaussian, with phi = -eta, beta = -beta_cp,
    D = epsilon (the definition file already stores the signed coefficients).
    """

    n: tuple = eqx.field(static=True)
    d: tuple = eqx.field(static=True)
    t: tuple = eqx.field(static=True)
    phi: tuple = eqx.field(static=True)
    beta: tuple = eqx.field(static=True)
    D: tuple = eqx.field(static=True)
    gamma: tuple = eqx.field(static=True)

    def contribution(self, delta, tau, namespace=jnp):
        total = 0.0 * delta * tau
        for n, d, t, phi, beta, D, gamma in zip(
            self.n, self.d, self.t, self.phi, self.beta, self.D, self.gamma
        ):
            dd = delta - D
            dt = tau - gamma
            exponent = phi * dd * dd + beta * dt * dt
            total = total + n * static_pow(delta, d) * static_pow(
                tau, t
            ) * namespace.exp(exponent)
        return total

    def derivatives(self, delta, tau) -> Tuple:
        out = [jnp.zeros_like(delta * tau)] * 6
        for n, d, t, phi, beta, D, gamma in zip(
            self.n, self.d, self.t, self.phi, self.beta, self.D, self.gamma
        ):
            dd = delta - D
            dt = tau - gamma
            base = (
                n
                * static_pow(delta, d)
                * static_pow(tau, t)
                * jnp.exp(phi * dd * dd + beta * dt * dt)
            )
            part = _from_log_derivatives(
                base,
                delta,
                tau,
                d,
                t,
                2.0 * phi * dd,
                jnp.full_like(delta, 2.0 * phi),
                2.0 * beta * dt,
                jnp.full_like(tau, 2.0 * beta),
            )
            out = [a + b for a, b in zip(out, part)]
        return tuple(out)


class GeneralizedExponential(ResidualTerm):
    """sum n * delta^d * tau^t * exp(-g * delta^l)

    Covers CoolProp ResidualHelmholtzExponential. Generalizes `ExponentialDensity`
    (which is the g = 1 case) with an explicit coefficient g in the
    exponential, the exponent 'l' in `ExponentialDensity` does not allow to
    make it a special case and required another class.

    NOTE :
    Terms with g = 0 are plain polynomials; the converter (in make_utils/eos.py)
    routes those to `Polynomial` instead
    """

    n: tuple = eqx.field(static=True)
    d: tuple = eqx.field(static=True)
    t: tuple = eqx.field(static=True)
    p: tuple = eqx.field(static=True)
    g: tuple = eqx.field(static=True)

    def contribution(self, delta, tau, namespace=jnp):
        groups: dict = {}
        for n, d, t, p, g in zip(self.n, self.d, self.t, self.p, self.g):
            groups.setdefault((g, p), []).append((n, d, t))

        total = 0.0 * delta * tau
        for (g, p), items in groups.items():
            inner = 0.0 * delta * tau
            for n, d, t in items:
                inner = inner + n * static_pow(delta, d) * static_pow(tau, t)
            total = total + inner * namespace.exp(-g * static_pow(delta, p))
        return total

    def derivatives(self, delta, tau) -> Tuple:
        out = [jnp.zeros_like(delta * tau)] * 6
        zero = jnp.zeros_like(delta)
        for n, d, t, p, g in zip(self.n, self.d, self.t, self.p, self.g):
            L_d = -g * p * static_pow(delta, p - 1)
            dL_d = -g * p * (p - 1) * static_pow(delta, p - 2)
            base = (
                n
                * static_pow(delta, d)
                * static_pow(tau, t)
                * jnp.exp(-g * static_pow(delta, p))
            )
            part = _from_log_derivatives(base, delta, tau, d, t, L_d, dL_d, zero, zero)
            out = [a + b for a, b in zip(out, part)]
        return tuple(out)


# - Following terms are less common and might pose numerical issues, they are to be dealt with
class NonAnalytic(ResidualTerm):
    """TODO  CoolProp ResidualHelmholtzNonAnalytic (Span-Wagner critical terms):

        sum n * Delta^b * delta * psi
        Delta = theta^2 + B*((delta-1)^2)^a
        theta = (1-tau) + A*((delta-1)^2)^(1/(2*beta))
        psi   = exp(-C*(delta-1)^2 - D*(tau-1)^2)

    Needed only for CO2 and water. It is quite delicate; fractional powers of
    (delta-1)^2 give infinite grad at delta = 1, so `contribution` will need a
    double-where guard, the one term that stresses differentiability at the
    critical point.

    Ref: Span & Wagner (1996) doi:10.1063/1.555991.
    """

    def contribution(self, delta, tau):
        raise NotImplementedError("NonAnalytic not implemented yet")


class GaoB(ResidualTerm):
    """TODO  CoolProp ResidualHelmholtzGaoB (Gao et al. generalized bell term).
    Needed for ammonia.

    Ref: Gao et al. (2020) doi:10.1063/5.0021459.
    """

    def contribution(self, delta, tau):
        raise NotImplementedError("GaoB not implemented yet")
