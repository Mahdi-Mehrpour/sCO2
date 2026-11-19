"""Cost model v2 for the split-flow sCO2 plant (objective: total cost rate).

Purpose: capture how cost changes with design and size, for optimisation. It is
NOT a quotation. Every parameter is in COST with its source or marked
placeholder; override any of them with cost_plant(..., p={...}).

Structure
  machines (turbine, compressor, gearbox, generator, motor): [W19] power laws,
       C = a * SP^b * f_T, 2017 USD (Weiland et al. 2019, GT2019-90493)
  recuperators: [W19] power law in UA (default) or [WDS16] linear in UA
  heaters: AREA based, A = UA / U_gas, cost = c_area * A * (1 + fouling) * f_T
  cooler: [WDS16] linear in UA (segmented UA, near-critical CO2)
  cooling tower: power law in heat rejected, calibrated to the earlier model
  installation and indirect costs as fractions of purchased cost
  total cost rate = (capital recovery + O&M) on the installed capital + water,
       per hour of operation (no revenue, no electricity price)
Sizes come from the frozen hardware (design_hardware) and the plant results."""
import math
import warnings

import numpy as np

K = 273.15
CEPCI = {2016: 541.7, 2017: 567.5, 2023: 797.9}

COST = {
    "cepci_target": 797.9,          # cost year 2023; update to the current index
    "recuperator_basis": "W19",     # "W19" (UA^0.7544) or "WDS16" (linear)
    "c_recup_WDS16": 2500.0,        # USD/(kW/K), 2016 USD, [WDS16] as in the old model
    "c_cooler": 1700.0,             # USD/(kW/K), 2016 USD, shell and tube, [WDS16]
    "c_heater_per_UA": 5.0,         # USD/(W/K) at U_ref, 2016 USD ([WDS16] fin tube)
    "U_ref": 40.0,                  # W/m2K, gives the same price per UA as the old model
    "U_gas": 40.0,                  # W/m2K, gas-side limited overall U (PLACEHOLDER 30-60)
    "fouling_margin": 0.20,         # extra heater area for dust/fouling (PLACEHOLDER)
    "tower_C0": 897.68e3,           # USD (2023) at Q0, calibrated to the earlier model
    "tower_Q0": 20.523e6,           # W
    "tower_exponent": 0.8,          # PLACEHOLDER (typical 0.6-0.8)
    "f_inst": 0.30,                 # installation, as in the earlier model
    "f_indirect": 0.0,              # engineering, ducts, dust, contingency (PLACEHOLDER;
                                    # the DLR study uses 0.67)
    "mult": {},                     # per-component cost multipliers, e.g. {"Turbine": 1.3}
    "r": 0.08, "N": 20,             # discount rate, life (settings_splitflow ECON_BASE)
    "om_fraction": 0.02,            # O&M per year, fraction of installed capital
    "water_price": 2.275,           # USD/m3 (PLACEHOLDER)
}

# [W19] Table 2: a, b, c (linear T coefficient), d (quadratic), valid range, break point
W19 = {
    "radial_turbine": (406_200, 0.8, 0.0, 1.137e-5, 8, 35),
    "ig_compressor": (1_230_000, 0.3992, 0.0, 0.0, 1.5, 200),
    "gearbox": (177_200, 0.2434, 0.0, 0.0, 4, 10),
    "generator": (108_900, 0.5463, 0.0, 0.0, 4, 750),
    "motor_synchronous": (211_400, 0.6227, 0.0, 0.0, 0.15, 15),
    "recuperator": (49.45, 0.7544, 0.02141, 0.0, 1.6e5, 2.15e8),
}
GAS_HEATER_D = 5.4e-5               # [W19] temperature coefficient of the gas heater
T_BP = 550.0                        # C, [W19] break point


def f_T(c, d, T_C):
    if T_C is None or T_C < T_BP:
        return 1.0
    dT = T_C - T_BP
    return 1.0 + c * dT + d * dT ** 2


def w19(kind, sp, T_C=None, warn=None):
    a, b, c, d, lo, hi = W19[kind]
    if not (lo <= sp <= hi) and warn is not None:
        warn.append(f"{kind}: SP = {sp:g} outside the [W19] range {lo:g}-{hi:g}")
    return a * sp ** b * f_T(c, d, T_C)


def crf(i, n):
    return 1.0 / n if i == 0 else i * (1 + i) ** n / ((1 + i) ** n - 1)


def cost_plant(P, hw, p=None, hours=None, annual_energy_MWh=None):
    """P: evaluate_plant() result; hw: design_hardware() result (UA in W/K).
    hours: operating hours per year (default P['hours']); annual_energy_MWh:
    simulated annual net electrical energy (for LCOE only, not for the cost rate)."""
    from .parasitics_splitflow import PARASITIC
    q = dict(COST)
    if p:
        q.update(p)
    res, par, cp = P["res"], P["par"], P["cooler"]
    st = res["states"]
    hours = float(hours if hours is not None else P["hours"])
    warn = []
    esc17 = q["cepci_target"] / CEPCI[2017]
    esc16 = q["cepci_target"] / CEPCI[2016]
    T7, T8 = st[7]["T"] - K, st[8]["T"] - K
    Tb = st[11]["T"] - K

    gb, gen, mot = PARASITIC["eta_gearbox"], PARASITIC["eta_generator"], PARASITIC["eta_motor"]
    W_T, W_C = par["W_turbine"] / 1e6, par["W_compressor"] / 1e6        # shaft, MW
    W_gen, W_mot = W_T * gb * gen, W_C / (gb * mot)                     # electrical, MW

    rows = {}                                    # name -> (size text, USD, source)
    rows["Turbine"] = (f"{W_T:.2f} MW shaft", w19("radial_turbine", W_T, T7, warn) * esc17,
                       "[W19] radial turbine")
    rows["Gearbox + generator"] = (
        f"{W_T:.2f} MW / {W_gen:.2f} MWe",
        (w19("gearbox", W_T, None, warn) + w19("generator", W_gen, None, warn)) * esc17,
        "[W19] gearbox + generator")
    rows["Compressor + motor"] = (
        f"{W_C:.2f} MW / {W_mot:.2f} MWe",
        (w19("ig_compressor", W_C, None, warn) + w19("motor_synchronous", W_mot, None, warn)) * esc17,
        "[W19] integrally geared compressor + synchronous motor")

    for name, ua, Tmax in (("HTR", hw["UA_HTR"], T8), ("LTR", hw["UA_LTR"], None)):
        if q["recuperator_basis"] == "W19":
            c = w19("recuperator", ua, Tmax, warn) * esc17
            src = "[W19] recuperator, UA^0.7544"
        else:
            c = q["c_recup_WDS16"] * ua / 1e3 * esc16
            src = "[WDS16] linear in UA"
        rows[name] = (f"UA {ua / 1e3:.0f} kW/K", c, src)

    c_area = q["c_heater_per_UA"] * q["U_ref"]                          # USD/m2 (2016)
    for name, ua, Tmax in (("Main heater", hw["UA_MH"], T7), ("Branch heater", hw["UA_BH"], Tb)):
        area = ua / q["U_gas"]
        c = c_area * area * (1 + q["fouling_margin"]) * f_T(0.0, GAS_HEATER_D, Tmax) * esc16
        rows[name] = (f"{area:.0f} m2 (UA {ua / 1e3:.0f} kW/K, U {q['U_gas']:g})", c,
                      "area based: UA/U x USD/m2 x (1 + fouling) x f_T")

    ua_c = cp["UA"]
    rows["Cooler"] = (f"UA {ua_c / 1e3:.0f} kW/K", q["c_cooler"] * ua_c / 1e3 * esc16,
                      "[WDS16] shell and tube, linear in segmented UA")
    Qc = cp["Q"]
    rows["Cooling tower"] = (f"{Qc / 1e6:.1f} MWth",
                             q["tower_C0"] * (Qc / q["tower_Q0"]) ** q["tower_exponent"]
                             * q["cepci_target"] / CEPCI[2023],
                             "power law in heat rejected, calibrated to the earlier model")

    mult = q["mult"]
    rows = {k: (s, c * mult.get(k, 1.0), src) for k, (s, c, src) in rows.items()}
    PEC = sum(c for _, c, _ in rows.values())
    TCI = PEC * (1 + q["f_inst"]) * (1 + q["f_indirect"])
    f_cap = (1 + q["f_inst"]) * (1 + q["f_indirect"])

    c_rec = crf(q["r"], q["N"])
    cap_year, om_year = c_rec * TCI, q["om_fraction"] * TCI
    water_m3 = cp["flow"] * 3600.0 * hours / 1000.0
    water_year = water_m3 * q["water_price"]
    annual = cap_year + om_year + water_year
    out = {
        "rows": rows, "PEC": PEC, "TCI": TCI, "CRF": c_rec, "hours": hours,
        "capital_per_year": cap_year, "om_per_year": om_year, "water_per_year": water_year,
        "annual_cost": annual, "cost_rate": annual / hours,          # USD/h of operation
        "Z_dot": {k: (c_rec + q["om_fraction"]) * c * f_cap / hours for k, (_, c, _) in rows.items()},
        "water_m3": water_m3, "net_el_design_MW": P["net_el"] / 1e6,
        "capital_per_kWe": TCI / (P["net_el"] / 1e3), "warnings": warn, "params": q,
    }
    if annual_energy_MWh and annual_energy_MWh > 0:
        out["annual_energy_MWh"] = annual_energy_MWh
        out["LCOE_sim"] = annual / annual_energy_MWh                 # USD/MWh, simulated energy
    return out
