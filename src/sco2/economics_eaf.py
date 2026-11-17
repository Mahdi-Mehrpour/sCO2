"""
sco2_eaf_economics.py
=====================
Economic and exergoeconomic (SPECO) model for the split-flow sCO2 cycle
(no recompressor) recovering EAF off-gas heat, with a wet cooling tower.

Input : your state table (CSV) plus a few scalars from your cycle model.
Output: purchased cost Z_k and cost rate Z_dot_k of every component, total capital
        investment, LCOE, NPV, IRR, simple and discounted payback period, and SPECO
        stream costs with exergoeconomic indicators (c_F, c_P, C_D, f, r).

Layout and state numbering (as in your model)
---------------------------------------------
 1 compressor inlet          2 compressor outlet / split point
 3 LTR cold outlet           5 mixer outlet = HTR cold inlet
 6 HTR cold outlet           7 turbine inlet (main heater outlet)
 8 turbine outlet            9 HTR hot outlet
10 LTR hot outlet = cooler inlet          11 branch heater outlet
Gas: g1 heater inlet -> (main heater) -> g2 -> (branch heater) -> g3 stack

REFERENCES (each equation is tagged)
------------------------------------
[W19]   Weiland, Lance, Pidaparti, "sCO2 power cycle component cost correlations from DOE data
        spanning multiple scales and applications", ASME GT2019-90493, doi:10.1115/GT2019-90493.
        C = a*SP^b*f_T, 2017 USD (CEPCI 567.5), Table 2 coefficients and uncertainty bands.
[WDS16] Wright, Davidson, Scammell, "Thermo-economic analysis of four sCO2 waste heat recovery
        power systems", 5th Int. sCO2 Power Cycles Symposium, San Antonio, 2016. Table 4:
        recuperator 2500, fin-tube heater 5000, shell-and-tube CO2 chiller 1700 $/(kW/K),
        UA from END-POINT LMTD; FOAK; whole-plant benchmark 1717-2019 $/kWe.
        (No cost year stated; 2016 USD assumed.)
[P19]   Pidaparti, White, O'Connell, Weiland, "Cooling system cost and performance models for
        economic sCO2 plant optimization with respect to cold sCO2 temperature", 3rd European
        sCO2 Conf., 2019, doi:10.17185/duepublico/48915. Modified Zanker wet-tower cost (1967 USD).
[M18]   Marchionni, Bianchi, Tassou, Energy 148 (2018) 1140-1152, doi:10.1016/j.energy.2018.02.005.
        Installation + auxiliaries +30 %, O&M 30 $/kWe-yr, escalation 3 %, degradation 1 %,
        PBP interpolation (Eq. 27).
[B20]   Biondi, Giovannelli, Di Lorenzo, Salvini, Energy Reports 6 (2020) 298-304,
        doi:10.1016/j.egyr.2020.11.147. Salvage 5 % of capital, taxes omitted for self-consumption.
[NREL95] Short, Packey, Holt, NREL/TP-462-5173 (1995): NPV, LCOE (discounted costs / discounted
        energy), payback.
[BTM96] Bejan, Tsatsaronis, Moran, Thermal Design and Optimization, Wiley 1996: CRF, levelized
        cost rates (Ch. 7), cost balances and exergoeconomic variables (Ch. 8).
[LT06]  Lazzaretto, Tsatsaronis, Energy 31 (2006) 1257-1289, doi:10.1016/j.energy.2005.03.011 (SPECO).
[CEPCI] Chemical Engineering Plant Cost Index: 1967 = 109.7, 2016 = 541.7, 2017 = 567.5,
        2023 = 797.9 (chemengonline.com). Update CEPCI_TARGET to your cost year.

Assumptions you should review are marked  # ASSUMPTION
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

CEPCI = {1967: 109.7, 2016: 541.7, 2017: 567.5, 2023: 797.9}
K = 3.6  # (USD/GJ) * MW -> USD/h


# =============================================================================
# 1. Inputs
# =============================================================================
@dataclass
class State:
    m: float      # kg/s
    T: float      # C
    P: float      # MPa
    h: float      # kJ/kg
    s: float      # kJ/kg.K
    ex: float     # kJ/kg  (physical exergy)

    @property
    def E(self) -> float:      # MW
        return self.m * self.ex / 1e3


def load_states(path: str) -> Dict[int, State]:
    """Read the state table your model writes (columns: State, Description, Mass fraction,
    Flow (kg/s), T (C), P (MPa), h (kJ/kg), s (kJ/kg.K), ex (kJ/kg))."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[int(row["State"])] = State(float(row["Flow (kg/s)"]), float(row["T (C)"]),
                                           float(row["P (MPa)"]), float(row["h (kJ/kg)"]),
                                           float(row["s (kJ/kg.K)"]), float(row["ex (kJ/kg)"]))
    return out


@dataclass
class CycleInputs:
    st: Dict[int, State]
    y: float                    # branch fraction
    T_g1: float                 # gas temperatures, C
    T_g2: float
    T_g3: float
    E_g1: float                 # gas physical exergy, MW
    E_g2: float
    E_g3: float
    W_fan_gas: float            # off-gas fan electric power, MW (from your model)
    # cooling (your wet-tower assumptions)
    T_wb: float = 18.0
    tower_approach: float = 4.0
    water_rise: float = 8.0
    tower_power_frac: float = 0.01          # fans + pumps / cooler duty  # ASSUMPTION (yours)
    makeup_factor: float = 1.33             # makeup / evaporation        # ASSUMPTION (yours)
    h_fg: float = 2442.0                    # kJ/kg at ~25 C
    makeup_kg_s: Optional[float] = None     # pass your model's value to override
    # parasitics
    eta_gb: float = 0.985
    eta_gen: float = 0.975
    eta_mot: float = 0.955
    # heat exchanger sizing: "endpoint" = UA from end-point LMTD,
    # "max" = larger of end-point and segmented UA (needs UA_seg)
    ua_basis: str = "max"
    UA_seg: Optional[Dict[str, float]] = None   # kW/K: HTR LTR MH BH cool


@dataclass
class EconInputs:
    cepci_target: float = CEPCI[2023]
    f_inst: float = 0.30          # installation + auxiliaries on purchased cost [M18]
    r: float = 0.05               # discount rate [WDS16], [M18]
    N: int = 20                   # years [WDS16], [M18]
    hours: float = 330 * 24 * 0.9  # 7128 h/yr (your operation assumptions)
    c_e: float = 0.08             # avoided electricity price, $/kWh  # ASSUMPTION (site tariff) [B20]
    c_OM: float = 30.0            # $/kWe-yr [M18]
    er: float = 0.03              # O&M escalation [M18]
    dr: float = 0.01              # output degradation [M18]
    salvage: float = 0.05         # of TCI [B20]
    water_price: float = 1.0      # $/m3 makeup water  # ASSUMPTION (site tariff)
    c_gas: float = 0.0            # $/GJ exergy of waste heat (free)  # ASSUMPTION
    T_bp_check: bool = True


# =============================================================================
# 2. Sizing from the state table
# =============================================================================
def lmtd(dTa: float, dTb: float) -> float:
    if dTa <= 0 or dTb <= 0:
        raise ValueError(f"non-positive terminal difference {dTa:.2f}, {dTb:.2f}")
    return dTa if abs(dTa - dTb) < 1e-9 else (dTa - dTb) / math.log(dTa / dTb)


def size(ci: CycleInputs) -> dict:
    s = ci.st
    m = s[7].m
    Tw_in = ci.T_wb + ci.tower_approach
    Tw_out = Tw_in + ci.water_rise
    d = {}
    d["W_T"] = m * (s[7].h - s[8].h) / 1e3
    d["W_C"] = s[1].m * (s[2].h - s[1].h) / 1e3
    d["Q_HTR"] = s[8].m * (s[8].h - s[9].h) / 1e3
    d["Q_LTR"] = s[9].m * (s[9].h - s[10].h) / 1e3
    d["Q_MH"] = m * (s[7].h - s[6].h) / 1e3
    d["Q_BH"] = s[11].m * (s[11].h - s[2].h) / 1e3
    d["Q_cool"] = s[10].m * (s[10].h - s[1].h) / 1e3
    # end-point LMTD (counter-flow), as required by the [WDS16] cost basis
    d["LMTD_HTR"] = lmtd(s[8].T - s[6].T, s[9].T - s[5].T)
    d["LMTD_LTR"] = lmtd(s[9].T - s[3].T, s[10].T - s[2].T)
    d["LMTD_MH"] = lmtd(ci.T_g1 - s[7].T, ci.T_g2 - s[6].T)
    d["LMTD_BH"] = lmtd(ci.T_g2 - s[11].T, ci.T_g3 - s[2].T)
    d["LMTD_cool"] = lmtd(s[10].T - Tw_out, s[1].T - Tw_in)
    for k in ("HTR", "LTR", "MH", "BH", "cool"):
        d[f"LMTD_ep_{k}"] = d[f"LMTD_{k}"]
        d[f"UA_ep_{k}"] = d[f"Q_{k}"] * 1e3 / d[f"LMTD_ep_{k}"]
        d[f"UA_{k}"] = d[f"UA_ep_{k}"]
        if ci.ua_basis == "max" and ci.UA_seg is not None:
            d[f"UA_{k}"] = max(d[f"UA_ep_{k}"], ci.UA_seg[k])
        d[f"LMTD_{k}"] = d[f"Q_{k}"] * 1e3 / d[f"UA_{k}"]          # kW/K
    # electricity
    d["W_mot"] = d["W_C"] / (ci.eta_gb * ci.eta_mot)
    d["W_eg"] = d["W_T"] * ci.eta_gb * ci.eta_gen
    d["W_tower"] = ci.tower_power_frac * d["Q_cool"]
    d["W_fan_gas"] = ci.W_fan_gas
    d["W_net"] = d["W_eg"] - d["W_mot"] - d["W_tower"] - ci.W_fan_gas
    evap = d["Q_cool"] * 1e3 / ci.h_fg
    d["makeup_kg_s"] = ci.makeup_kg_s if ci.makeup_kg_s is not None else ci.makeup_factor * evap
    d["Tw_in"], d["Tw_out"] = Tw_in, Tw_out
    return d


# =============================================================================
# 3. Purchased equipment cost (each in its own base year)
# =============================================================================
W19 = {  # a, b, c, d, (min, max), (unc-, unc+)
    "radial_turbine": (406_200, 0.8, 0.0, 1.137e-5, (8, 35), (-0.32, 0.51)),
    "ig_compressor": (1_230_000, 0.3992, 0.0, 0.0, (1.5, 200), (-0.40, 0.48)),
    "gearbox": (177_200, 0.2434, 0.0, 0.0, (4, 10), (-0.15, 0.20)),
    "generator": (108_900, 0.5463, 0.0, 0.0, (4, 750), (-0.19, 0.23)),
    "motor_synchronous": (211_400, 0.6227, 0.0, 0.0, (0.15, 15), (-0.15, 0.20)),
    "recuperator": (49.45, 0.7544, 0.02141, 0.0, (1.6e5, 2.15e8), (-0.31, 0.38)),
}


def w19(kind: str, sp: float, T_max: Optional[float] = None, warn: list = None) -> float:
    """C = a*SP^b*f_T,  f_T = 1 + c*dT + d*dT^2 above 550 C  [W19 Eq. 3-4]"""
    a, b, c, dd, (lo, hi), _ = W19[kind]
    if warn is not None and not lo <= sp <= hi:
        warn.append(f"{kind}: SP={sp:.4g} outside [W19] range {lo:g}-{hi:g}")
    dT = max((T_max or 0) - 550.0, 0.0)
    return a * sp ** b * (1 + c * dT + dd * dT ** 2)


def zanker_1967(Q_MW: float, T_wb_C: float, approach_K: float, range_K: float) -> float:
    """Modified Zanker wet-tower construction cost, 1967 USD [P19]:
       C = Q0*(Q/Q0)^0.8426 / (C_wb*A0*(A/A0)^0.4376 + 39.2*R - 586)
       C_wb = 279 / (1 + 0.0335*(85 - T_wb)^1.143);  Q in Btu/h, A, R, T_wb in F.
    Reference Q0 = 3465 MMBtu/h, A0 = 8.52 F. NOTE: formula reconstructed from the PDF text;
    check against the NETL spreadsheet. Small duties are an extrapolation."""
    Q = Q_MW * 1e6 * 3.412142
    Q0, A0 = 3465e6, 8.52
    A, R, Twb = approach_K * 1.8, range_K * 1.8, T_wb_C * 1.8 + 32
    Cwb = 279 / (1 + 0.0335 * (85 - Twb) ** 1.143)
    return Q0 * (Q / Q0) ** 0.8426 / (Cwb * A0 * (A / A0) ** 0.4376 + 39.2 * R - 586)


def equipment_costs(ci: CycleInputs, d: dict, e: EconInputs) -> dict:
    """Purchased equipment cost in target-year USD, with low/high bands and source."""
    warn = []
    esc = lambda yr: e.cepci_target / CEPCI[yr]
    s = ci.st
    rows = {}

    def add(name, cost, yr, band, src):
        rows[name] = dict(PEC=cost * esc(yr), lo=cost * esc(yr) * (1 + band[0]),
                          hi=cost * esc(yr) * (1 + band[1]), source=src)

    add("Turbine", w19("radial_turbine", d["W_T"], s[7].T, warn), 2017,
        W19["radial_turbine"][5], "[W19] radial turbine, W_sh")
    add("Gearbox+generator",
        w19("gearbox", d["W_T"], None, warn) + w19("generator", d["W_eg"], None, warn), 2017,
        (-0.19, 0.23), "[W19] gearbox (W_T) + generator (W_e)")
    add("Compressor+motor",
        w19("ig_compressor", d["W_C"], None, warn) + w19("motor_synchronous", d["W_mot"], None, warn),
        2017, W19["ig_compressor"][5], "[W19] IG compressor + synchronous motor")
    add("HTR", 2500 * d["UA_HTR"], 2016, (-0.30, 0.30), "[WDS16] 2500 $/(kW/K), end-point UA")
    add("LTR", 2500 * d["UA_LTR"], 2016, (-0.30, 0.30), "[WDS16] 2500 $/(kW/K), end-point UA")
    add("Main heater", 5000 * d["UA_MH"], 2016, (-0.20, 1.00), "[WDS16] 5000 $/(kW/K), fin tube")
    add("Branch heater", 5000 * d["UA_BH"], 2016, (-0.20, 1.00), "[WDS16] 5000 $/(kW/K), fin tube")
    add("Cooler", 1700 * d["UA_cool"], 2016, (-0.30, 0.30), "[WDS16] 1700 $/(kW/K), shell & tube")
    add("Cooling tower", zanker_1967(d["Q_cool"], ci.T_wb, ci.tower_approach, ci.water_rise), 1967,
        (-0.30, 0.30), "[P19] modified Zanker (band ASSUMED, not in source)")
    # cross-check only: Weiland recuperator correlation
    xchk = {k: w19("recuperator", d[f"UA_{k}"] * 1e3, s[8].T, warn) * esc(2017) for k in ("HTR", "LTR")}
    return {"rows": rows, "warnings": warn, "xcheck_W19_recup": xchk}


# =============================================================================
# 4. Economics: TCI, cash flows, LCOE, NPV, IRR, PBP
# =============================================================================
def crf(i: float, n: int) -> float:
    """CRF = i(1+i)^n/((1+i)^n - 1)  [BTM96], [NREL95]"""
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


def cash_flows(TCI, W_net_MW, water_m3_yr, e: EconInputs):
    """Year-by-year (k = 1..N) energy, revenue and expenses  [M18 Eq. 24, 26], [B20]."""
    rows = []
    for k in range(1, e.N + 1):
        E_k = W_net_MW * e.hours * (1 - e.dr) ** k                    # MWh
        rev = E_k * 1e3 * e.c_e                                        # $
        om = W_net_MW * 1e3 * e.c_OM * (1 + e.er) ** k                 # $
        wat = water_m3_yr * e.water_price * (1 + e.er) ** k            # $  (escalated like O&M)
        sal = e.salvage * TCI if k == e.N else 0.0
        rows.append(dict(k=k, E=E_k, rev=rev, om=om, water=wat, salvage=sal,
                         cf=rev - om - wat + sal))
    return rows


def indicators(TCI, flows, e: EconInputs) -> dict:
    disc = lambda k: (1 + e.r) ** k
    pv_cost = TCI + sum((f["om"] + f["water"] - f["salvage"]) / disc(f["k"]) for f in flows)
    pv_E = sum(f["E"] / disc(f["k"]) for f in flows)
    LCOE = pv_cost / pv_E                                               # $/MWh  [NREL95]
    NPV = -TCI + sum(f["cf"] / disc(f["k"]) for f in flows)             # [NREL95]

    def payback(discounted: bool) -> float:                             # [M18 Eq. 27]
        cum = -TCI
        for f in flows:
            cf = f["cf"] / (disc(f["k"]) if discounted else 1)
            if cum + cf >= 0:
                return f["k"] - 1 + (-cum) / cf
            cum += cf
        return math.inf

    def irr() -> float:
        g = lambda x: -TCI + sum(f["cf"] / (1 + x) ** f["k"] for f in flows)
        lo, hi = -0.99, 10.0
        if g(lo) * g(hi) > 0:
            return float("nan")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if g(mid) > 0 else (lo, mid)
        return 0.5 * (lo + hi)

    return dict(LCOE=LCOE, NPV=NPV, IRR=irr(), PBP=payback(False), DPBP=payback(True))


# =============================================================================
# 5. Levelized component cost rates Z_dot_k  [BTM96 Ch. 7]
# =============================================================================
def cost_rates(rows: dict, TCI: float, d: dict, flows, e: EconInputs, water_m3_yr: float) -> dict:
    """Z_dot_k = (CRF * TCI_k + levelized O&M_k) / hours  ($/h).
    O&M is allocated in proportion to TCI_k; makeup water goes to the cooling tower."""
    CRF = crf(e.r, e.N)
    disc = lambda k: (1 + e.r) ** k
    lev = lambda key: CRF * sum(f[key] / disc(f["k"]) for f in flows)   # levelized annual $
    OM_lev, W_lev = lev("om"), lev("water")
    pec_sum = sum(r["PEC"] for r in rows.values())
    out = {}
    for name, r in rows.items():
        tci_k = r["PEC"] * (1 + e.f_inst)
        share = r["PEC"] / pec_sum
        annual = CRF * tci_k * (1 - e.salvage / (1 + e.r) ** e.N) + share * OM_lev
        out[name] = annual / e.hours
    out["_water_cost_rate"] = W_lev / e.hours
    return out


# =============================================================================
# 6. SPECO exergoeconomic balance  [LT06], [BTM96 Ch. 8]
# =============================================================================
def speco(ci: CycleInputs, d: dict, Zd: dict, e: EconInputs) -> dict:
    """Unknowns: C1 C2 C3 C5 C6 C7 C8 C9 C10 C11 Cg1 Cg2 Cg3 C_WT ($/h) and c_el ($/GJ).
    Physical exergy is not split into thermal/mechanical parts (simplification of [LT06]).
    The cooler + tower are dissipative: their cost rates, tower power and makeup water are
    charged to stream 1, so they end up in the cost of electricity (one common treatment
    of dissipative components [BTM96 Ch. 8])."""
    s, y = ci.st, ci.y
    E = {i: s[i].E for i in s}
    Eg1, Eg2, Eg3 = ci.E_g1, ci.E_g2, ci.E_g3
    idx = {n: i for i, n in enumerate(
        ["C1", "C2", "C3", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "Cg1", "Cg2", "Cg3", "CWT", "cel"])}
    A, b = np.zeros((15, 15)), np.zeros(15)

    def eq(row, coeffs, rhs):
        for n, v in coeffs.items():
            A[row, idx[n]] += v
        b[row] = rhs

    eq(0, {"C7": 1, "C8": -1, "CWT": -1}, -Zd["Turbine"])                       # turbine
    eq(1, {"C7": 1 / E[7], "C8": -1 / E[8]}, 0)                                 # F rule
    eq(2, {"CWT": 1, "cel": -K * d["W_eg"]}, -Zd["Gearbox+generator"])          # generator
    eq(3, {"C1": 1, "C2": -1, "cel": K * d["W_mot"]}, -Zd["Compressor+motor"])  # compressor
    eq(4, {"C8": 1, "C5": 1, "C9": -1, "C6": -1}, -Zd["HTR"])                   # HTR
    eq(5, {"C8": 1 / E[8], "C9": -1 / E[9]}, 0)                                 # F rule
    eq(6, {"C9": 1, "C2": 1 - y, "C10": -1, "C3": -1}, -Zd["LTR"])              # LTR
    eq(7, {"C9": 1 / E[9], "C10": -1 / E[10]}, 0)                               # F rule
    eq(8, {"C3": 1, "C11": 1, "C5": -1}, 0)                                     # mixer
    eq(9, {"Cg1": 1, "C6": 1, "Cg2": -1, "C7": -1}, -Zd["Main heater"])         # main heater
    eq(10, {"Cg1": 1 / Eg1, "Cg2": -1 / Eg2}, 0)                                # F rule
    eq(11, {"Cg2": 1, "C2": y, "Cg3": -1, "C11": -1}, -Zd["Branch heater"])     # branch heater
    eq(12, {"Cg2": 1 / Eg2, "Cg3": -1 / Eg3}, 0)                                # F rule
    eq(13, {"C10": 1, "C1": -1, "cel": K * d["W_tower"]},
       -(Zd["Cooler"] + Zd["Cooling tower"] + Zd["_water_cost_rate"]))          # cooler+tower
    eq(14, {"Cg1": 1, "cel": -K * d["W_fan_gas"]}, K * e.c_gas * Eg1)          # gas inlet + fan
    x = np.linalg.solve(A, b)
    C = {n: x[i] for n, i in idx.items()}
    c_el = C["cel"]

    # overall balance check: all costs in = electricity out + stack loss
    Z_all = sum(v for k, v in Zd.items() if not k.startswith("_")) + Zd["_water_cost_rate"]
    resid = Z_all + K * e.c_gas * Eg1 - (c_el * K * d["W_net"] + C["Cg3"])

    def ind(name, Fx, Px, CF, CP, Z):
        cF = CF / (K * Fx) if Fx else float("nan")
        cP = CP / (K * Px) if Px else float("nan")
        ED = Fx - Px
        CD = cF * ED * K
        return dict(component=name, F_MW=Fx, P_MW=Px, ED_MW=ED, eps=Px / Fx, c_F=cF, c_P=cP,
                    Z_dot=Z, C_D=CD, Z_plus_CD=Z + CD, f=Z / (Z + CD) if Z + CD else float("nan"),
                    r=(cP - cF) / cF if cF else float("nan"))

    E2L, E2B = (1 - y) * E[2], y * E[2]
    C2L, C2B = (1 - y) * C["C2"], y * C["C2"]
    tab = [
        ind("Turbine", E[7] - E[8], d["W_T"], C["C7"] - C["C8"], C["CWT"], Zd["Turbine"]),
        ind("Gearbox+generator", d["W_T"], d["W_eg"], C["CWT"], c_el * K * d["W_eg"],
            Zd["Gearbox+generator"]),
        ind("Compressor+motor", d["W_mot"], E[2] - E[1], c_el * K * d["W_mot"], C["C2"] - C["C1"],
            Zd["Compressor+motor"]),
        ind("HTR", E[8] - E[9], E[6] - E[5], C["C8"] - C["C9"], C["C6"] - C["C5"], Zd["HTR"]),
        ind("LTR", E[9] - E[10], E[3] - E2L, C["C9"] - C["C10"], C["C3"] - C2L, Zd["LTR"]),
        ind("Main heater", Eg1 - Eg2, E[7] - E[6], C["Cg1"] - C["Cg2"], C["C7"] - C["C6"],
            Zd["Main heater"]),
        ind("Branch heater", Eg2 - Eg3, E[11] - E2B, C["Cg2"] - C["Cg3"], C["C11"] - C2B,
            Zd["Branch heater"]),
    ]
    streams = {n: (C[n], C[n] / (K * (E[int(n[1:])] if n[1:].isdigit() else 1)))
               for n in C if n.startswith("C") and n[1:].isdigit()}
    return dict(c_el_GJ=c_el, c_el_MWh=c_el * 3.6, C=C, streams=streams, table=tab,
                balance_residual=resid, mixer_ED=E[3] + E[11] - E[5],
                cooler_exergy_loss=E[10] - E[1])


# =============================================================================
# 7. Driver and report
# =============================================================================
def run(ci: CycleInputs, e: EconInputs = EconInputs()) -> dict:
    d = size(ci)
    eq = equipment_costs(ci, d, e)
    rows = eq["rows"]
    PEC = sum(r["PEC"] for r in rows.values())
    TCI = PEC * (1 + e.f_inst)
    water_m3_yr = d["makeup_kg_s"] * 3600 * e.hours / 1000
    flows = cash_flows(TCI, d["W_net"], water_m3_yr, e)
    res = indicators(TCI, flows, e)
    # capital-cost uncertainty bands
    for tag in ("lo", "hi"):
        tci_b = sum(r[tag] for r in rows.values()) * (1 + e.f_inst)
        res[f"{tag}"] = indicators(tci_b, cash_flows(tci_b, d["W_net"], water_m3_yr, e), e)
        res[f"{tag}"]["TCI"] = tci_b
    Zd = cost_rates(rows, TCI, d, flows, e, water_m3_yr)
    sp = speco(ci, d, Zd, e)
    return dict(size=d, costs=eq, PEC=PEC, TCI=TCI, water_m3_yr=water_m3_yr, flows=flows,
                econ=res, Z_dot=Zd, speco=sp)


def report(R: dict) -> str:
    d, rows, ec, sp = R["size"], R["costs"]["rows"], R["econ"], R["speco"]
    L = ["# sCO2 EAF waste-heat unit: economic and exergoeconomic results", ""]
    L.append("## Sizes")
    L.append(f"W_T {d['W_T']:.3f} MW | W_C {d['W_C']:.3f} MW | W_el,gross {d['W_eg']:.3f} | "
             f"motor {d['W_mot']:.3f} | tower {d['W_tower']:.3f} | gas fan {d['W_fan_gas']:.3f} | "
             f"**W_net {d['W_net']:.3f} MW**")
    L.append("| HX | Q (MW) | effective LMTD (K) | UA (kW/K) |\n|---|---|---|---|")
    for k in ("HTR", "LTR", "MH", "BH", "cool"):
        L.append(f"| {k} | {d['Q_'+k]:.3f} | {d['LMTD_'+k]:.2f} | {d['UA_'+k]:.1f} |")
    L += ["", "## Purchased equipment cost and cost rates",
          "| Component | PEC (k$) | band (k$) | Z_dot ($/h) | Source |", "|---|---|---|---|---|"]
    for n, r in rows.items():
        L.append(f"| {n} | {r['PEC']/1e3:,.0f} | {r['lo']/1e3:,.0f}-{r['hi']/1e3:,.0f} | "
                 f"{R['Z_dot'][n]:.1f} | {r['source']} |")
    L.append(f"| Makeup water | | | {R['Z_dot']['_water_cost_rate']:.1f} | {R['water_m3_yr']:,.0f} m3/yr |")
    L.append(f"\nPEC {R['PEC']/1e6:.2f} M$ | **TCI {R['TCI']/1e6:.2f} M$** "
             f"({R['TCI']/(d['W_net']*1e3):,.0f} $/kWe net; [WDS16] FOAK benchmark 1717-2019 $/kWe)")
    x = R["costs"]["xcheck_W19_recup"]
    L.append(f"Cross-check [W19] recuperator: HTR {x['HTR']/1e3:,.0f} k$, LTR {x['LTR']/1e3:,.0f} k$")
    L += ["", "## Economic indicators (capital band in brackets)",
          f"- LCOE: **{ec['LCOE']:.1f} $/MWh** [{ec['lo']['LCOE']:.1f}-{ec['hi']['LCOE']:.1f}]",
          f"- NPV: **{ec['NPV']/1e6:.2f} M$** [{ec['hi']['NPV']/1e6:.2f} to {ec['lo']['NPV']/1e6:.2f}]",
          f"- IRR: **{ec['IRR']*100:.1f} %** [{ec['hi']['IRR']*100:.1f}-{ec['lo']['IRR']*100:.1f}]",
          f"- Simple payback: **{ec['PBP']:.2f} yr** [{ec['lo']['PBP']:.2f}-{ec['hi']['PBP']:.2f}]",
          f"- Discounted payback: **{ec['DPBP']:.2f} yr** [{ec['lo']['DPBP']:.2f}-{ec['hi']['DPBP']:.2f}]",
          "", "## SPECO",
          f"Unit cost of electricity c_el = {sp['c_el_GJ']:.2f} $/GJ = **{sp['c_el_MWh']:.1f} $/MWh** "
          f"(levelized, no degradation). Balance residual {sp['balance_residual']:.2e} $/h.",
          "| Component | E_F (MW) | E_P (MW) | E_D (MW) | eps | c_F ($/GJ) | c_P ($/GJ) | "
          "Z_dot ($/h) | C_D ($/h) | f (%) | r (%) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in sp["table"]:
        L.append(f"| {t['component']} | {t['F_MW']:.3f} | {t['P_MW']:.3f} | {t['ED_MW']:.3f} | "
                 f"{t['eps']:.3f} | {t['c_F']:.2f} | {t['c_P']:.2f} | {t['Z_dot']:.1f} | "
                 f"{t['C_D']:.1f} | {t['f']*100:.1f} | {t['r']*100:.1f} |")
    L.append(f"\nMixer E_D {sp['mixer_ED']:.4f} MW; cooler exergy loss {sp['cooler_exergy_loss']:.3f} MW "
             f"(cost charged to the product).")
    L.append("\n| Stream | C ($/h) | c ($/GJ) |\n|---|---|---|")
    for n, (Cn, cn) in sp["streams"].items():
        L.append(f"| {n[1:]} | {Cn:.1f} | {cn:.2f} |")
    L.append(f"| gas stack (g3) | {sp['C']['Cg3']:.1f} | |")
    if R["costs"]["warnings"]:
        L += ["", "## Validity warnings"] + [f"- {w}" for w in R["costs"]["warnings"]]
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    # ---------------------------------------------------------------------------------
    # TEST DATA: your earlier 30 MPa state table (states as you sent them). Replace with
    # load_states("your_states.csv") and the gas/fan values from your current model.
    # ---------------------------------------------------------------------------------
    raw = {1: (55.55, 32.0, 7.7, 306.234, 1.34631, 214.97),
           2: (55.55, 76.5, 29.9987, 343.424, 1.36233, 247.384),
           3: (40.12, 278.69, 29.6987, 682.35, 2.14379, 353.318),
           5: (55.55, 297.43, 29.6987, 707.289, 2.18823, 365.006),
           6: (55.55, 377.31, 29.4018, 810.917, 2.36017, 417.37),
           7: (55.55, 550.0, 28.8137, 1029.52, 2.66224, 545.911),
           8: (55.55, 398.0, 7.9357, 865.139, 2.68978, 373.319),
           9: (55.55, 307.43, 7.8563, 761.511, 2.52584, 318.571),
           10: (55.55, 96.31, 7.7778, 516.685, 2.00087, 230.264),
           11: (15.42, 347.29, 29.6987, 772.174, 2.29727, 397.381)}
    st = load_states(sys.argv[1]) if len(sys.argv) > 1 else {k: State(*v) for k, v in raw.items()}
    # gas exergy: inlet 11.5425 MW; E_g2 and E_g3 from heater exergy balances of that run
    ci = CycleInputs(st=st, y=0.2776, T_g1=800.0, T_g2=397.3052, T_g3=159.3782,
                     E_g1=11.5425, E_g2=3.6127, E_g3=0.6246, W_fan_gas=0.131)
    R = run(ci)
    txt = report(R)
    print(txt)
    with open("sco2_eaf_results.md", "w") as f:
        f.write(txt)
