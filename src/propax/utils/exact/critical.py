"""
The critical point, is where the correlation's own pressure surface goes flat:

    dP/drho|_T = 0        d2P/drho2|_T = 0

Rather than taking the declared T_crit and rho_crit from CoolProp files we find
them here in high precision to ensure perfect coherence in the package.
"""

from typing import NamedTuple, Tuple

import mpmath as mp

from .precision import PRECISION

_MAX_STEPS = 60
_MAX_REL_STEP = 0.05


class ReducedEOS:
    """
    EOS's alphar and the pressure, in (delta, tau) with numerical derivatives
    in reduced form and extended precision for offline computation.

    `psi = delta * Z` is the pressure without its dimensional factor:

        P = rho_red * R_u * T * psi(delta, tau)

    so at fixed T the derivatives of psi in delta are the derivatives of P in
    rho, up to that constant. The criticality conditions are `psi_d = psi_dd = 0`.
    """

    def __init__(self, eos, dps: int):
        self.eos = eos
        self.dps = dps
        mp.mp.dps = dps
        self.T_red = mp.mpf(float(eos.T_red))
        self.rho_red_mol = mp.mpf(float(eos.rho_red_mol))
        self.R_u = mp.mpf(float(eos.R_u))
        self.molar_mass = mp.mpf(float(eos.molar_mass))

    def a(self, d, t):
        return self.eos.alphar(d, t, mp)

    def a_d(self, d, t, n: int = 1):
        return mp.diff(lambda x: self.a(x, t), d, n)

    def psi(self, d, t):
        """psi = delta * (delta * da/ddelta + 1)"""
        return d + d * d * self.a_d(d, t, 1)

    def psi_d(self, d, t):
        return 1 + 2 * d * self.a_d(d, t, 1) + d * d * self.a_d(d, t, 2)

    def psi_dd(self, d, t):
        return (
            2 * self.a_d(d, t, 1)
            + 4 * d * self.a_d(d, t, 2)
            + d * d * self.a_d(d, t, 3)
        )

    def psi_ddd(self, d, t):
        return (
            6 * self.a_d(d, t, 2)
            + 6 * d * self.a_d(d, t, 3)
            + d * d * self.a_d(d, t, 4)
        )

    def gibbs(self, d, t):
        """
        ln f = alphar + delta a_delta + ln(rho_mol R T), and at fixed T only
        ln(delta) survives of the last term so equal Gibbs energy is this
        function taking the same value on both branches.

        tau only ideal terms are cancelled analytically instead of numerically
        (which can cause numerical noise).
        """
        return self.a(d, t) + d * self.a_d(d, t, 1) + mp.log(d)

    def gibbs_d(self, d, t):
        return mp.diff(lambda x: self.gibbs(x, t), d)


class CriticalPoint(NamedTuple):
    delta_c: mp.mpf
    tau_c: mp.mpf
    T_crit: mp.mpf
    rho_crit_mol: mp.mpf
    P_crit: mp.mpf
    B: mp.mpf
    """
    We prove in /docs/superancillaries.md that delta_L,V = delta_c +- B sqrt(theta/(1-theta))
    """


def solve_critical_point(eos) -> CriticalPoint:
    """Newton on `psi_d = psi_dd = 0`, starting from the reducing point (provided as
    an EOS parameter).

    /!\\ Reducing parameters are calibrated coefficients while critical point is a mathematical
    limit of the parametrized EOS.
    """
    r = ReducedEOS(eos, PRECISION.dps)
    d = mp.mpf(float(eos.rho_crit_mol)) / r.rho_red_mol
    t = r.T_red / mp.mpf(float(eos.T_crit))
    tol = PRECISION.newton_tol(PRECISION.dps)

    for _ in range(_MAX_STEPS):
        F = mp.matrix([r.psi_d(d, t), r.psi_dd(d, t)])
        J = mp.matrix(2, 2)
        J[0, 0] = mp.diff(
            lambda x: r.psi_d(x, t), d
        )  # evaluates at delta (d_psi/d_delta)_tau
        J[0, 1] = mp.diff(lambda y: r.psi_d(d, y), t)
        J[1, 0] = mp.diff(lambda x: r.psi_dd(x, t), d)
        J[1, 1] = mp.diff(lambda y: r.psi_dd(d, y), t)
        s = mp.lu_solve(J, -F)
        s[0] = _clip(s[0], _MAX_REL_STEP * abs(d))
        s[1] = _clip(s[1], _MAX_REL_STEP * abs(t))
        d, t = d + s[0], t + s[1]
        if max(abs(s[0] / d), abs(s[1] / t)) < tol:
            break
    else:
        raise RuntimeError(
            f"the criticality conditions did not converge in {_MAX_STEPS} steps"
        )

    psi_ddd = r.psi_ddd(d, t)
    psi_dt = mp.diff(lambda y: r.psi_d(d, y), t)
    if psi_ddd <= 0 or psi_dt >= 0:
        raise RuntimeError(
            f"the critical point has psi_ddd = {mp.nstr(psi_ddd, 6)} and "
            f"psi_dtau = {mp.nstr(psi_dt, 6)}; the dome does not open below it"
        )

    T_crit = r.T_red / t  # tau is T_red / T_crit
    return CriticalPoint(
        delta_c=d,
        tau_c=t,
        T_crit=T_crit,
        rho_crit_mol=d * r.rho_red_mol,
        P_crit=r.rho_red_mol * r.R_u * T_crit * r.psi(d, t),
        B=mp.sqrt(-6 * psi_dt * t / psi_ddd),
    )


def _clip(x, cap):
    return mp.sign(x) * cap if abs(x) > cap else x


def saturated_pair(
    eos,
    crit: CriticalPoint,
    theta,
    seed=None,
) -> Tuple:
    """The saturated pair at T = T_crit * (1 - theta).
    Returns `(delta_L, delta_V, tau)`, all mpf.
    """
    reduced = ReducedEOS(eos, PRECISION.working_dps(theta))
    th = mp.mpf(theta)
    t = crit.tau_c / (1 - th)
    if seed is None:
        s = crit.B * mp.sqrt(th / (1 - th))
        dL, dV = crit.delta_c + s, crit.delta_c - s
    else:
        dL, dV = mp.mpf(seed[0]), mp.mpf(seed[1])
    tol = PRECISION.newton_tol(PRECISION.dps)

    # the vapour branch spans ten decades down to the triple point -> take the log
    yL, yV = mp.log(dL), mp.log(dV)
    for _ in range(_MAX_STEPS):
        dL, dV = mp.exp(yL), mp.exp(yV)
        F = mp.matrix(
            [
                reduced.psi(dL, t) - reduced.psi(dV, t),  # residual psi
                reduced.gibbs(dL, t)
                - reduced.gibbs(dV, t),  # residual gibbs free energy
            ]
        )
        J = mp.matrix(2, 2)
        J[0, 0], J[0, 1] = reduced.psi_d(dL, t) * dL, -reduced.psi_d(dV, t) * dV
        J[1, 0], J[1, 1] = reduced.gibbs_d(dL, t) * dL, -reduced.gibbs_d(dV, t) * dV
        step = mp.lu_solve(J, -F)
        yL, yV = yL + step[0], yV + step[1]
        # A step that leaves the reals means the iterate has wandered where the
        # fractional tau exponents are no longer defined
        if mp.im(yL) != 0 or mp.im(yV) != 0:
            raise RuntimeError(
                f"the saturated pair left the reals at theta = {mp.nstr(th, 6)}; the "
                "seed is too far off "
                + (
                    "(the expansion does not reach this theta)"
                    if seed is None
                    else "(the continuation step is too large)"
                )
            )
        if max(abs(step[0]), abs(step[1])) < tol:  # already relative, in logs
            break
    else:
        raise RuntimeError(
            f"the saturated pair did not converge at theta = {mp.nstr(th, 6)}"
        )

    dL, dV = mp.exp(yL), mp.exp(yV)
    if not dL > dV:
        raise RuntimeError(
            f"the saturated pair collapsed onto the trivial root at theta = "
            f"{mp.nstr(th, 6)}: delta_L = {mp.nstr(dL, 12)}, "
            f"delta_V = {mp.nstr(dV, 12)}"
        )
    return dL, dV, t


def _extrapolated_seed(last, before, theta):
    """Strategy for extrapolating the seed on and away the critical point.

    Entries are `(s, delta_L, ln delta_V)`, s = sqrt(theta): the variables in
    which both ordinates are linear near T_crit, so a secant needs no regime
    test. See /docs/superancillaries.md.
    """
    s1, dL1, yV1 = last
    s0, dL0, yV0 = before
    s = mp.sqrt(mp.mpf(theta))
    w = (s - s1) / (s1 - s0)
    return dL1 + w * (dL1 - dL0), mp.exp(yV1 + w * (yV1 - yV0))


def saturated_pairs(eos, crit: CriticalPoint, thetas):
    """The saturated pair at each theta, continued from the one before.

    `thetas` ascends: the first two are anchored on the critical expansion,
    whose error is O(theta) and so is negligible where the walk starts, and
    every solve after is seeded with a point that rides the line through its
    two predecessors.

    Returns four lists of mpf tau, delta_L, delta_V, and the reduced pressure psi
    in the order given.
    """
    r_by_theta = {}
    out_t, out_L, out_V, out_psi = [], [], [], []
    history: list = []

    for theta in thetas:
        seed = (
            _extrapolated_seed(history[-1], history[-2], theta)
            if len(history) >= 2
            else None
        )
        dL, dV, tau = saturated_pair(eos, crit, theta, seed)
        history.append((mp.sqrt(mp.mpf(theta)), dL, mp.log(dV)))
        work = PRECISION.working_dps(theta)
        r = r_by_theta.setdefault(work, ReducedEOS(eos, work))
        mp.mp.dps = work
        out_t.append(tau)
        out_L.append(dL)
        out_V.append(dV)
        # pressure retrieved in vapor phase (less prone to error)
        out_psi.append(r.psi(dV, tau))

    return out_t, out_L, out_V, out_psi
