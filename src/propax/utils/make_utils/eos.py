"""
/!\\ : How to read CoolProp files

To add a term, read in the CoolProp sources files (github.com/CoolProp/CoolProp):
- include/CoolProp/fluids/Helmholtz.h : read the term's class definition with its docstring
- src/Backends/Helmholtz/Fluids/FluidLibrary.h : read its block in parse_alphar or
  parse_alpha0, it shows the JSON names each parameter
- src/Helmholtz.cpp writes its derivatives
"""

import math

from .source import require_coolprop

# --------------------------------------------- ideal-gas blocks
# handlers mutate the accumulator {a1, a2, a_log, pe_n, pe_v}


def _lead_or_offset(blk, acc):
    acc["a1"] += blk["a1"]
    acc["a2"] += blk["a2"]


def _log_tau(blk, acc):
    # additive so it composes with the CP0 forms, which also contribute a ln(tau)
    # term; a_log starts at 0 and a normal fluid has a single LogTau block
    acc["a_log"] += blk["a"]


# Tau and delta both designate the reduced temperature and reduced density
# CoolProp gives some fluids' ideal part as cp0(T) rather than as alpha0 terms :
# With alpha0 = h0/(RT) - 1 - s0/R and cv0/R = -tau^2 alpha0_tau_tau = cp0/R - 1,
# a term cp0/R = c T^t integrates (relative to the reference T0, tau0 = Tc/T0) to
#
#   t != 0, -1:  alpha0 += -c Tc^t/(t(t+1)) tau^-t          (a power-in-tau term)
#                         - c T0^(t+1)/((t+1) Tc) tau        (-> a2, linear)
#                         + c T0^t / t                       (-> a1, constant)
#   t == 0    :  alpha0 += c ln(tau) - (c/tau0) tau + c(1 - ln tau0)
#
# The power-in-tau terms are delta-independent, so they ride the same d=0
# residual-polynomial route as IdealGasHelmholtzPower. P, cv and cp depend only
# on a_log and these powers (reference-independent); a1/a2 set the energy datum,
# alongside the fluid's separate EnthalpyEntropyOffset block. t == -1 would give
# a tau*ln(tau) term the schema has no form for, so it is rejected explicitly.


def _cp0_constant(blk, acc):
    """cp0/R = c (IdealGasHelmholtzCP0Constant)."""
    c, tau0 = blk["cp_over_R"], blk["Tc"] / blk["T0"]
    acc["a_log"] += c
    acc["a2"] += -c / tau0
    acc["a1"] += c * (1.0 - math.log(tau0))


def _cp0_polyT(blk, acc):
    """cp0/R = sum c_i T^t_i (IdealGasHelmholtzCP0PolyT)."""
    Tc, T0 = blk["Tc"], blk["T0"]
    tau0 = Tc / T0
    for c, t in zip(blk["c"], blk["t"]):
        if t == 0:
            acc["a_log"] += c
            acc["a2"] += -c / tau0
            acc["a1"] += c * (1.0 - math.log(tau0))
        elif t == -1:
            raise ValueError(
                "IdealGasHelmholtzCP0PolyT term with t = -1 integrates to "
                "tau*ln(tau), which the schema has no ideal term for"
            )
        else:
            acc["_tau_pow"].append((-c * Tc**t / (t * (t + 1)), -t))
            acc["a2"] += -c * T0 ** (t + 1) / ((t + 1) * Tc)
            acc["a1"] += c * T0**t / t


def _planck_einstein_T(blk, acc):
    acc["pe_n"] += list(blk["n"])
    acc["pe_v"] += list(blk["v"])  # already in Kelvin


def _planck_einstein(blk, acc):
    """Same Einstein term, but theta is stored already reduced (theta = v/T_c).

    The schema keeps pe_v in Kelvin (the runtime divides by T_c), so multiply
    back: pe_v = t * T_c.
    """
    T_c = acc["_T_c"]
    acc["pe_n"] += list(blk["n"])
    acc["pe_v"] += [t * T_c for t in blk["t"]]


def _ideal_power(blk, acc):
    acc["_tau_pow"] += list(zip(blk["n"], blk["t"]))


IDEAL_BLOCK_HANDLERS = {
    "IdealGasHelmholtzLead": _lead_or_offset,
    "IdealGasHelmholtzEnthalpyEntropyOffset": _lead_or_offset,
    "IdealGasHelmholtzLogTau": _log_tau,
    "IdealGasHelmholtzPlanckEinsteinFunctionT": _planck_einstein_T,
    "IdealGasHelmholtzPlanckEinstein": _planck_einstein,
    "IdealGasHelmholtzPower": _ideal_power,
    "IdealGasHelmholtzCP0Constant": _cp0_constant,
    "IdealGasHelmholtzCP0PolyT": _cp0_polyT,
}


# handlers mutate the accumulator holding the schema's coefficient arrays


def _power(blk, acc):
    for n, t, d, lp in zip(blk["n"], blk["t"], blk["d"], blk["l"]):
        if lp == 0:
            acc["poly_n"].append(n)
            acc["poly_t"].append(t)
            acc["poly_d"].append(float(d))
        else:
            acc["exp_n"].append(n)
            acc["exp_t"].append(t)
            acc["exp_d"].append(float(d))
            acc["exp_p"].append(float(lp))


def _gaussian(blk, acc):
    for n, t, d, eta, beta, gamma, eps in zip(
        blk["n"],
        blk["t"],
        blk["d"],
        blk["eta"],
        blk["beta"],
        blk["gamma"],
        blk["epsilon"],
    ):
        acc["gauss_n"].append(n)
        acc["gauss_t"].append(t)
        acc["gauss_d"].append(float(d))
        acc["gauss_phi"].append(-eta)  # propax stores exp(+phi (d-D)^2 ...)
        acc["gauss_beta"].append(-beta)
        acc["gauss_gamma"].append(gamma)
        acc["gauss_D"].append(eps)


def _exponential(blk, acc):
    for n, t, d, g, p in zip(blk["n"], blk["t"], blk["d"], blk["g"], blk["l"]):
        if g == 0:
            acc["poly_n"].append(n)
            acc["poly_t"].append(t)
            acc["poly_d"].append(float(d))
        else:
            acc["gexp_n"].append(n)
            acc["gexp_t"].append(t)
            acc["gexp_d"].append(float(d))
            acc["gexp_g"].append(float(g))
            acc["gexp_p"].append(float(p))


RESIDUAL_BLOCK_HANDLERS = {
    "ResidualHelmholtzPower": _power,
    "ResidualHelmholtzGaussian": _gaussian,
    "ResidualHelmholtzExponential": _exponential,
}


def convert_eos(cp: dict, cp_name: str) -> tuple[dict, dict, dict]:
    """Returns (constants, ideal, residual) schema dicts."""
    CP = require_coolprop()
    eos = cp["EOS"][0]

    red = eos["STATES"]["reducing"]

    ideal_acc = dict(
        a1=0.0, a2=0.0, a_log=0.0, pe_n=[], pe_v=[], _T_c=red["T"], _tau_pow=[]
    )
    for blk in eos["alpha0"]:
        handler = IDEAL_BLOCK_HANDLERS.get(blk["type"])
        if handler is None:
            raise ValueError(
                f"{cp_name}: unsupported ideal-gas block '{blk['type']}' "
                f"(supported: {sorted(IDEAL_BLOCK_HANDLERS)})"
            )
        handler(blk, ideal_acc)

    residual = {
        k: []
        for k in (
            "poly_n",
            "poly_t",
            "poly_d",
            "exp_n",
            "exp_t",
            "exp_d",
            "exp_p",
            "gauss_n",
            "gauss_t",
            "gauss_d",
            "gauss_phi",
            "gauss_beta",
            "gauss_gamma",
            "gauss_D",
            "gexp_n",
            "gexp_t",
            "gexp_d",
            "gexp_g",
            "gexp_p",
        )
    }
    for blk in eos["alphar"]:
        handler = RESIDUAL_BLOCK_HANDLERS.get(blk["type"])
        if handler is None:
            raise ValueError(
                f"{cp_name}: unsupported residual block '{blk['type']}' "
                f"(supported: {sorted(RESIDUAL_BLOCK_HANDLERS)})"
            )
        handler(blk, residual)

    # ideal power terms are delta-independent -> emitted as d = 0 polynomials
    ideal_acc.pop("_T_c")
    for n, t in ideal_acc.pop("_tau_pow"):  # type: ignore
        residual["poly_n"].append(n)
        residual["poly_t"].append(t)
        residual["poly_d"].append(0.0)

    consts = dict(
        R_u=eos["gas_constant"],
        molar_mass=eos["molar_mass"],
        T_red=red["T"],
        P_red=red["p"],
        rho_red_mol=red["rhomolar"],
        T_triple=CP.PropsSI("Ttriple", cp_name),
        P_triple=CP.PropsSI("ptriple", cp_name),
        # stated upper validity of the EOS
        T_max=eos["T_max"],
        P_max=eos["p_max"],
    )
    return consts, ideal_acc, residual
