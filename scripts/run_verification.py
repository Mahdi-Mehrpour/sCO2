"""Verification checks of the split-flow plant model (saved design).
Writes tables/13_verification.csv. These checks show that the code does what
is intended (balances, limiting cases, numerical independence, definitions).
They are not a validation against measurements."""
import inspect
import json
from pathlib import Path

import cantera as ct
import pandas as pd
from CoolProp.CoolProp import PropsSI

from sco2.cooling_splitflow import cooler_performance
from sco2.cycle_adv import splitflow_adv
from sco2.economics_eaf import run
from sco2.economics_splitflow import build_inputs, fast_economics
from sco2.exergy_splitflow import exergy_plant
from sco2.heat_source_adv import EAF, GasStream, couple_cascade
from sco2.plant_splitflow import evaluate_plant, get_gas
from sco2.settings_splitflow import ECON_BASE

K = 273.15
FL = "CO2"
T0 = 298.15
dTp, Tmin = EAF["dT_pinch_heater"], EAF["T_stack_min"]

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
D = {k: v.default for k, v in inspect.signature(splitflow_adv).parameters.items()}
DP0 = {k: 0.0 for k in D if k.startswith("dp_")}      # all pressure drops zero
CASE = {"y": opt["y"], "T_b": opt["T_b"], "pr": opt["pr"]}
gas = get_gas()

P = evaluate_plant(**CASE)
res, hs, par, cp = P["res"], P["hs"], P["par"], P["cooler"]
st, y, m = res["states"], res["y"], hs["m_co2"]
mg, T_in = P["m_gas"], P["T_in"]

_cache = {}


def cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def cycle(**kw):
    """Cycle of the saved design, with optional overrides."""
    return splitflow_adv(**{"T1": P["T1"], **CASE, **kw})


rows = []


def test(category, name, fn):
    """fn returns (value, criterion text, passed). An exception gives SKIPPED."""
    try:
        value, criterion, passed = fn()
        result = "PASS" if passed else "FAIL"
    except Exception as exc:
        value, criterion = float("nan"), f"{type(exc).__name__}: {str(exc)[:90]}"
        result = "SKIPPED"
    rows.append({"No": len(rows) + 1, "Category": category, "Check": name,
                 "Value": float(f"{value:.6g}"), "Criterion": criterion,
                 "Result": result})


def within(value, limit, unit=""):
    return value, f"|value| < {limit:g} {unit}".strip(), abs(value) < limit


# ======================================================================
# Properties
# ======================================================================
test("Properties", "State temperatures recomputed from (h, p), worst case (K)",
     lambda: within(max(abs(PropsSI("T", "H", s["h"], "P", s["p"], FL) - s["T"])
                        for s in st.values()), 1e-4, "K"))
test("Properties", "State entropies recomputed from (T, p), worst case (J/kg.K)",
     lambda: within(max(abs(PropsSI("S", "T", s["T"], "P", s["p"], FL) - s["s"])
                        for s in st.values()), 1e-2, "J/kg.K"))


def t_crit():
    Tc, pc = PropsSI("Tcrit", FL), PropsSI("pcrit", FL)
    ok = abs(Tc - 304.1282) < 0.01 and abs(pc - 7.3773e6) < 2e3
    return Tc, "critical point 304.128 K and 7.3773 MPa (value shown: Tcrit in K)", ok


def t_super():
    mT = st[1]["T"] - PropsSI("Tcrit", FL)
    mp = (st[1]["p"] - PropsSI("pcrit", FL)) / 1e6
    return mT, f"compressor inlet above the critical point (pressure margin {mp:.2f} MPa); a small margin is a design risk", mT > 0 and mp > 0


test("Properties", "CO2 critical constants (CoolProp)", t_crit)
test("Properties", "Compressor inlet temperature above critical (K)", t_super)

# ======================================================================
# Balances
# ======================================================================
test("Balances", "Cycle energy balance, q_in + q_b - w_net - q_out (J/kg)",
     lambda: within(res["q_in"] + res["q_b"] - res["w_net"] - res["q_out"], 0.01, "J/kg"))
test("Balances", "Mixer energy balance (J/kg)",
     lambda: within(st[5]["h"] - ((1 - y) * st[3]["h"] + y * st[11]["h"]), 0.01, "J/kg"))
test("Balances", "HTR hot-side duty minus cold-side duty (J/kg)",
     lambda: within((st[8]["h"] - st[9]["h"]) - (st[6]["h"] - st[5]["h"]), 0.01, "J/kg"))
test("Balances", "LTR hot-side duty minus cold-side duty (J/kg)",
     lambda: within((st[9]["h"] - st[10]["h"]) - (1 - y) * (st[3]["h"] - st[2]["h"]),
                    0.01, "J/kg"))


def t_coupling():
    Qg = mg * (float(gas.h(T_in)) - float(gas.h(hs["T_out"])))
    d = (Qg - hs["Q"]) / hs["Q"]
    return d, "gas heat equals heat to the CO2 (relative) < 1e-4", abs(d) < 1e-4


test("Balances", "Heat from the gas minus heat to the CO2 (relative)", t_coupling)

ex = exergy_plant(P, gas, mg)
test("Balances", "System exergy balance (W)", lambda: within(ex["balance_W"], 10, "W"))
closure = (par["net_shaft"] - par["loss_gearbox"] - par["loss_generator"]
           - par["loss_motor"] - cp["P_aux"] - par["fan_gas"] - P["net_el"])
test("Balances", "Power balance, shaft power - losses - fans - net electrical (W)",
     lambda: within(closure, 1, "W"))

# ======================================================================
# Second law and bounds
# ======================================================================


def t_sgen():
    s_in, s_mid, s_out = (float(gas.s(T)) for T in (T_in, hs["T_gas_mid"], hs["T_out"]))
    g = {"Main compressor": m * (st[2]["s"] - st[1]["s"]),
         "Turbine": m * (st[8]["s"] - st[7]["s"]),
         "HTR": m * ((st[9]["s"] - st[8]["s"]) + (st[6]["s"] - st[5]["s"])),
         "LTR": m * ((st[10]["s"] - st[9]["s"]) + (1 - y) * (st[3]["s"] - st[2]["s"])),
         "Mixer": m * (st[5]["s"] - ((1 - y) * st[3]["s"] + y * st[11]["s"])),
         "Main heater": mg * (s_mid - s_in) + m * (st[7]["s"] - st[6]["s"]),
         "Branch heater": mg * (s_out - s_mid) + m * y * (st[11]["s"] - st[2]["s"])}
    name = min(g, key=g.get)
    return g[name], f"entropy generation >= 0 in all 7 components (smallest: {name}; value in W/K)", g[name] > -1e-3


test("Second law", "Smallest entropy generation of any component (W/K)", t_sgen)
test("Second law", "Cycle efficiency below Carnot between T1 and T7",
     lambda: (res["eta"], f"< Carnot {1 - st[1]['T'] / st[7]['T']:.3f}",
              res["eta"] < 1 - st[1]["T"] / st[7]["T"]))
test("Second law", "Net electrical power / available heat below Carnot (gas inlet, dead state)",
     lambda: (P["net_el"] / hs["Q_max"], f"< Carnot {1 - T0 / T_in:.3f}",
              P["net_el"] / hs["Q_max"] < 1 - T0 / T_in))
test("Second law", "System exergy efficiency between 0 and 1",
     lambda: (P["net_el"] / hs["ex_in"], "0 < value < 1",
              0 < P["net_el"] / hs["ex_in"] < 1))

# ======================================================================
# Definitions recomputed independently from CoolProp
# ======================================================================


def t_eta_c():
    h2s = PropsSI("H", "P", st[2]["p"], "S", st[1]["s"], FL)
    e = (h2s - st[1]["h"]) / (st[2]["h"] - st[1]["h"]) - D["eta_mc"]
    return within(e, 1e-6)


def t_eta_t():
    h8s = PropsSI("H", "P", st[8]["p"], "S", st[7]["s"], FL)
    e = (st[7]["h"] - st[8]["h"]) / (st[7]["h"] - h8s) - D["eta_t"]
    return within(e, 1e-6)


def eff(hot, cold, m_cold, q):
    q_hot = hot["h"] - PropsSI("H", "T", cold["T"], "P", hot["p"], FL)
    q_cold = m_cold * (PropsSI("H", "T", hot["T"], "P", cold["p"], FL) - cold["h"])
    return q / min(q_hot, q_cold)


test("Definitions", "Compressor isentropic efficiency minus input", t_eta_c)
test("Definitions", "Turbine isentropic efficiency minus input", t_eta_t)
test("Definitions", "HTR effectiveness minus input",
     lambda: within(eff(st[8], st[5], 1.0, st[8]["h"] - st[9]["h"]) - D["eps_htr"], 1e-6))
test("Definitions", "LTR effectiveness minus input",
     lambda: within(eff(st[9], st[2], 1 - y, st[9]["h"] - st[10]["h"]) - D["eps_ltr"], 1e-6))


def t_endpoints():
    errs = []
    for name, (c_in, c_out, h_out, h_in) in {"HTR": (5, 6, 9, 8), "LTR": (2, 3, 10, 9)}.items():
        p = res["pinch"][name]
        errs += [abs(p["T_cold"][0] - st[c_in]["T"]), abs(p["T_cold"][-1] - st[c_out]["T"]),
                 abs(p["T_hot"][0] - st[h_out]["T"]), abs(p["T_hot"][-1] - st[h_in]["T"])]
    return within(max(errs), 1e-3, "K")


test("Definitions", "Segmented recuperator profiles meet the cycle states, worst end point (K)",
     t_endpoints)

# ======================================================================
# Limiting cases
# ======================================================================


def t_ideal():
    s0 = cycle(eta_mc=1.0, eta_t=1.0, check_pinch=False, **DP0)["states"]
    e = max(abs(s0[2]["s"] - s0[1]["s"]), abs(s0[8]["s"] - s0[7]["s"]))
    return within(e, 1e-3, "J/kg.K")


def t_pressures():
    s0 = cycle(check_pinch=False, **DP0)["states"]
    hi = [s0[k]["p"] for k in (2, 3, 5, 6, 7, 11)]
    lo = [s0[k]["p"] for k in (8, 9, 10, 1)]
    return within(max(max(hi) - min(hi), max(lo) - min(lo)), 1e-3, "Pa")


def t_eps(which):
    s1 = cycle(eps_htr=1.0, eps_ltr=1.0, check_pinch=False, **DP0)["states"]
    if which == "HTR":
        e = min(abs(s1[9]["T"] - s1[5]["T"]), abs(s1[6]["T"] - s1[8]["T"]))
    else:
        e = min(abs(s1[10]["T"] - s1[2]["T"]), abs(s1[3]["T"] - s1[9]["T"]))
    return within(e, 1e-3, "K")


test("Limiting cases", "Ideal machines: entropy change across compressor and turbine (J/kg.K)", t_ideal)
test("Limiting cases", "Zero pressure drops: spread of pressures within each level (Pa)", t_pressures)
test("Limiting cases", "HTR effectiveness 1: one stream reaches the other's inlet T (K)",
     lambda: t_eps("HTR"))
test("Limiting cases", "LTR effectiveness 1: one stream reaches the other's inlet T (K)",
     lambda: t_eps("LTR"))

# ======================================================================
# Numerical independence
# ======================================================================


def t_seg(kind):
    r2 = cached("r120", lambda: cycle(n_seg=120))
    if kind == "UA":
        d = max(abs(r2["pinch"][n]["UA"] / res["pinch"][n]["UA"] - 1) for n in ("HTR", "LTR"))
        return d, "UA changes by < 1 % (30 to 120 segments)", d < 0.01
    d = max(abs(r2["pinch"][n]["min_dT"] - res["pinch"][n]["min_dT"]) for n in ("HTR", "LTR"))
    return d, "minimum temperature difference changes by < 0.5 K", d < 0.5


def t_heaters():
    h2 = couple_cascade(res, gas, T_in, mg, dTp, dTp, Tmin, n_seg=80)
    d = h2["m_co2"] / m - 1
    return d, "CO2 flow changes by < 0.5 % (20 to 80 heater segments)", abs(d) < 0.005


def t_cooler():
    d = cooler_performance("wet_tower", res, hs, n=120)["UA"] / cp["UA"] - 1
    return d, "cooler UA changes by < 1 % (30 to 120 segments)", abs(d) < 0.01


def t_gas_table():
    fine = GasStream(EAF["X"], EAF["p"], T0=T0, n=3000)
    h2 = couple_cascade(res, fine, T_in, mg, dTp, dTp, Tmin)
    d = h2["m_co2"] / m - 1
    return d, "CO2 flow changes by < 0.1 % (600 to 3000 gas table points)", abs(d) < 1e-3


test("Numerical", "Recuperator UA, relative change with 120 segments", lambda: t_seg("UA"))
test("Numerical", "Recuperator minimum temperature difference, change with 120 segments (K)",
     lambda: t_seg("dT"))
test("Numerical", "CO2 flow, relative change with 80 heater segments", t_heaters)
test("Numerical", "Cooler UA, relative change with 120 segments", t_cooler)
test("Numerical", "CO2 flow, relative change with a 5 times finer gas table", t_gas_table)

# ======================================================================
# Gas model
# ======================================================================
NAMES = {"N2": "Nitrogen", "CO2": "CO2", "H2O": "Water", "O2": "Oxygen",
         "CO": "CarbonMonoxide"}
test("Gas model", "Mole fractions add up to 1",
     lambda: within(sum(EAF["X"].values()) - 1.0, 1e-9))


def t_molar():
    M = sum(x * PropsSI("M", NAMES[s]) for s, x in EAF["X"].items()) * 1000.0
    return gas.M - M, f"molar mass {gas.M:.4f} kg/kmol equals the CoolProp value {M:.4f} (< 0.01)", abs(gas.M - M) < 0.01


test("Gas model", "Mixture molar mass, Cantera minus CoolProp (kg/kmol)", t_molar)

g = ct.Solution("gri30.yaml")
WINDOW = {"N2": (400.0, 1073.15), "CO2": (400.0, 1073.15), "H2O": (400.0, 1073.15),
          "O2": (400.0, 1073.15), "CO": (400.0, 480.0)}       # CO: CoolProp range is short


def t_species(sp):
    Ta, Tb = WINDOW[sp]
    g.TPX = Ta, 101325.0, {sp: 1.0}
    ha = g.enthalpy_mass
    g.TPX = Tb, 101325.0, {sp: 1.0}
    hb = g.enthalpy_mass
    ref = (PropsSI("H", "T", Tb, "P", 101325.0, NAMES[sp])
           - PropsSI("H", "T", Ta, "P", 101325.0, NAMES[sp]))
    d = (hb - ha) / ref - 1
    return d, f"enthalpy change {Ta:.0f} to {Tb:.0f} K within 1 % of CoolProp", abs(d) < 0.01


for sp in WINDOW:
    test("Gas model", f"{sp} enthalpy change, Cantera / CoolProp - 1",
         lambda sp=sp: t_species(sp))

# ======================================================================
# Economics
# ======================================================================


def econ():
    ci, e = build_inputs(P, ECON_BASE)
    return ci, e, run(ci, e)


def t_irr():
    ci, e, R = cached("econ", econ)
    irr = R["econ"]["IRR"]
    npv = -R["TCI"] + sum(f["cf"] / (1 + irr) ** f["k"] for f in R["flows"])
    return within(npv, 10, "USD")


def t_fast():
    ci, e, R = cached("econ", econ)
    return within(fast_economics(ci, e)["NPV"] - R["econ"]["NPV"], 1, "USD")


test("Economics", "NPV evaluated at the IRR (USD)", t_irr)
test("Economics", "SPECO cost balance residual (USD/h)",
     lambda: within(cached("econ", econ)[2]["speco"]["balance_residual"], 1e-6, "USD/h"))
test("Economics", "Total capital investment minus purchased cost x (1 + installation) (USD)",
     lambda: within(cached("econ", econ)[2]["TCI"]
                    - cached("econ", econ)[2]["PEC"] * (1 + cached("econ", econ)[1].f_inst), 1, "USD"))
test("Economics", "Net power of the economic module minus the plant model (MW)",
     lambda: within(cached("econ", econ)[2]["size"]["W_net"] - P["net_el"] / 1e6, 1e-6, "MW"))
test("Economics", "NPV of the fast path minus the full path (USD)", t_fast)

# ======================================================================
# Reproducibility, consistency with the tables, trends
# ======================================================================
test("Reproducibility", "Net electrical power, difference between two runs (W)",
     lambda: within(evaluate_plant(**CASE)["net_el"] - P["net_el"], 1e-9, "W"))


def t_table():
    t = pd.read_csv("tables/04_plant_performance.csv")
    v = pd.to_numeric(t.loc[t["Parameter"] == "Net electrical power", "Value"]).iloc[0]
    return within(v - P["net_el"] / 1e6, 1e-3, "MW")


test("Reproducibility", "Net electrical power, table 04 minus the model (MW)", t_table)


def t_flow():
    half = evaluate_plant(**CASE, check_pinch=False, m_gas=0.5 * mg)
    d = half["net_el"] / P["net_el"] - 0.5
    return d, "net power is proportional to the gas flow (< 1e-6)", abs(d) < 1e-6


def t_trend():
    v = [evaluate_plant(**CASE, check_pinch=False, T_in=Tc + K)["net_el"]
         for Tc in (T_in - K - 100, T_in - K, T_in - K + 100)]
    d = v[1] / P["net_el"] - 1
    return d, "net power rises with gas temperature (design -100 K < design < design +100 K); the middle value equals the design", (v[0] < v[1] < v[2]) and abs(d) < 1e-9


test("Trends", "Net power with half the gas flow / design - 0.5", t_flow)
test("Trends", "Net power at the design gas temperature / design - 1, with ordering at +-100 K", t_trend)

# ======================================================================
# Output
# ======================================================================
df = pd.DataFrame(rows)
Path("tables").mkdir(exist_ok=True)
df.to_csv("tables/13_verification.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 95)
print(df.to_string(index=False))
n = df["Result"].value_counts()
print(f"\n{len(df)} checks: {n.get('PASS', 0)} passed, {n.get('FAIL', 0)} failed, "
      f"{n.get('SKIPPED', 0)} skipped")
print(df.groupby("Category")["Result"].value_counts().unstack(fill_value=0).to_string())
print("\nSaved: tables/13_verification.csv")