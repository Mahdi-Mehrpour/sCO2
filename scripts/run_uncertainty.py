"""Uncertainty analysis of the split-flow plant at the saved design.
(1) One-at-a-time ranges (tornado) for every input; (2) Monte Carlo over all
inputs together (Latin hypercube, uniform ranges). The design variables
(y, T_b, pr) stay fixed; only inputs vary. All ranges are judgement values:
edit them in the add(...) lines below."""
import copy
import inspect
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc

import sco2.plant_splitflow as plant
from sco2.cooling_splitflow import COOLING
from sco2.cycle_adv import splitflow_adv
from sco2.economics_eaf import cash_flows, equipment_costs, indicators, size
from sco2.economics_splitflow import build_inputs
from sco2.heat_source_adv import EAF
from sco2.operation_splitflow import OPERATION
from sco2.parasitics_splitflow import PARASITIC
from sco2.plant_splitflow import evaluate_plant
from sco2.settings_splitflow import ECON_BASE

K = 273.15
N_MC = 300            # Monte Carlo samples (use 30 for a quick test)
SEED = 1

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
CASE = {"y": opt["y"], "T_b": opt["T_b"], "pr": opt["pr"]}
D = {k: v.default for k, v in inspect.signature(splitflow_adv).parameters.items()}
DP_KEYS = [k for k in D if k.startswith("dp_")]

# ---------------- state handling: change inputs, always restore -------------
SNAP = {"EAF": copy.deepcopy(dict(EAF)), "PAR": dict(PARASITIC),
        "WET": dict(COOLING["wet_tower"]),
        "OP": {k: OPERATION[k] for k in ("days_per_year", "plant_availability")}}
DIRTY = [False]


def reset():
    EAF.update(copy.deepcopy(SNAP["EAF"]))
    PARASITIC.update(SNAP["PAR"])
    COOLING["wet_tower"].update(SNAP["WET"])
    OPERATION.update(SNAP["OP"])
    if DIRTY[0]:
        plant._GAS.clear()
        DIRTY[0] = False


# ---------------- economics with component cost multipliers -----------------
def economics(Pl, e_over, mult, extras):
    """NPV etc. for the plant Pl. mult: cost multiplier per component;
    extras: list of unmodelled-capital fractions to evaluate."""
    ci, e = build_inputs(Pl, {**ECON_BASE, **e_over})
    d = size(ci)
    rows = equipment_costs(ci, d, e)["rows"]
    pec = sum(r["PEC"] * mult.get(n, 1.0) for n, r in rows.items())
    water = d["makeup_kg_s"] * 3600.0 * e.hours / 1000.0
    out = []
    for x in extras:
        tci = pec * (1.0 + e.f_inst) * (1.0 + x)
        out.append((indicators(tci, cash_flows(tci, d["W_net"], water, e), e), tci))
    return out


# ---------------- base case --------------------------------------------------
P0 = evaluate_plant(**CASE)
ci0, e0 = build_inputs(P0, ECON_BASE)
rows0 = equipment_costs(ci0, size(ci0), e0)["rows"]

# ---------------- input definitions -----------------------------------------
PARAMS = []


def add(name, group, unit, low, base, high, fn, basis="judgement"):
    PARAMS.append(dict(name=name, group=group, unit=unit, low=low, base=base,
                       high=high, fn=fn, basis=basis))


def cyc(key):
    return lambda v, c: c["cycle"].__setitem__(key, v)


def ecn(key, cast=float):
    return lambda v, c: c["econ"].__setitem__(key, cast(v))


def dct(d, key, off=0.0):
    return lambda v, c: d.__setitem__(key, v + off)


def set_dp(v, c):
    for k in DP_KEYS:
        c["cycle"][k] = D[k] * v


def set_h2o(v, c):
    X = dict(SNAP["EAF"]["X"])
    X["N2"] = X["N2"] - (v - X["H2O"])
    X["H2O"] = v
    EAF["X"] = X
    plant._GAS.clear()
    DIRTY[0] = True


WT = COOLING["wet_tower"]
add("Main compressor isentropic efficiency", "Cycle", "-", 0.80, D["eta_mc"], 0.88, cyc("eta_mc"))
add("Turbine isentropic efficiency", "Cycle", "-", 0.86, D["eta_t"], 0.93, cyc("eta_t"))
add("HTR effectiveness", "Cycle", "-", 0.86, D["eps_htr"], 0.94, cyc("eps_htr"))
add("LTR effectiveness", "Cycle", "-", 0.86, D["eps_ltr"], 0.94, cyc("eps_ltr"))
add("Pressure drops, scale on all six", "Cycle", "x", 0.5, 1.0, 2.0, set_dp)

add("Gas inlet temperature", "Heat source", "C", 700.0, EAF["T_in"] - K, 900.0,
    lambda v, c: c.__setitem__("T_in", v + K))
add("Gas mass flow", "Heat source", "kg/s", 15.0, EAF["m_gas"], 35.0,
    lambda v, c: c.__setitem__("m_gas", v))
add("Minimum stack temperature", "Heat source", "C", 120.0, EAF["T_stack_min"] - K, 180.0,
    dct(EAF, "T_stack_min", K))
add("Heater minimum temperature difference", "Heat source", "K", 15.0,
    EAF["dT_pinch_heater"], 30.0, dct(EAF, "dT_pinch_heater"))
add("Gas water fraction (N2 balances)", "Heat source", "-", 0.04, EAF["X"]["H2O"], 0.18, set_h2o)

add("Design wet-bulb temperature", "Cooling", "C", 12.0, WT["T_wb_C"], 26.0, dct(WT, "T_wb_C"))
add("Tower approach", "Cooling", "K", 3.0, WT["tower_approach"], 7.0, dct(WT, "tower_approach"))
add("Cooler approach (CO2 outlet - water inlet)", "Cooling", "K", 3.0, WT["approach"], 8.0,
    dct(WT, "approach"))
add("Cooling water temperature rise", "Cooling", "K", 6.0, WT["coolant_dT"], 12.0,
    dct(WT, "coolant_dT"))
add("Tower fan and pump, fraction of cooler duty", "Cooling", "-", 0.005,
    WT["fan_pump_fraction"], 0.02, dct(WT, "fan_pump_fraction"))

add("Gearbox efficiency", "Parasitics", "-", 0.97, PARASITIC["eta_gearbox"], 0.99,
    dct(PARASITIC, "eta_gearbox"))
add("Generator efficiency", "Parasitics", "-", 0.96, PARASITIC["eta_generator"], 0.985,
    dct(PARASITIC, "eta_generator"))
add("Compressor motor efficiency", "Parasitics", "-", 0.94, PARASITIC["eta_motor"], 0.97,
    dct(PARASITIC, "eta_motor"))
add("Gas-side pressure drop (off-gas fan)", "Parasitics", "Pa", 1500.0,
    PARASITIC["gas_dp_total"], 6000.0, dct(PARASITIC, "gas_dp_total"))
add("Off-gas fan efficiency", "Parasitics", "-", 0.60, PARASITIC["eta_fan_gas"], 0.80,
    dct(PARASITIC, "eta_fan_gas"))

add("Furnace operating days per year", "Operation", "days", 280.0,
    OPERATION["days_per_year"], 345.0, dct(OPERATION, "days_per_year"))
add("Availability of the sCO2 unit", "Operation", "-", 0.80,
    OPERATION["plant_availability"], 0.97, dct(OPERATION, "plant_availability"))

add("Discount rate", "Economics", "1/yr", 0.05, e0.r, 0.12, ecn("r"))
add("Electricity value", "Economics", "USD/kWh", 0.05, e0.c_e, 0.12, ecn("c_e"))
add("O&M cost", "Economics", "USD/kWe-yr", 20.0, e0.c_OM, 60.0, ecn("c_OM"))
add("Installation factor", "Economics", "-", 0.20, e0.f_inst, 0.50, ecn("f_inst"))
add("Economic life", "Economics", "yr", 15, e0.N, 25, ecn("N", lambda v: int(round(v))))
add("Makeup water price", "Economics", "USD/m3", 0.5, e0.water_price, 3.0, ecn("water_price"))

for name, r in rows0.items():
    add(f"Cost of {name.lower()}", "Cost", "x", r["lo"] / r["PEC"], 1.0, r["hi"] / r["PEC"],
        (lambda n: lambda v, c: c["mult"].__setitem__(n, v))(name),
        "band of the cost correlation (assumed for heaters and tower)")
add("Unmodelled capital (ducts, heat buffer, engineering)", "Cost", "fraction", 0.0, 0.0, 0.5,
    lambda v, c: c.__setitem__("extra", v), "not in the base case")


# ---------------- one evaluation of the whole plant -------------------------
def evaluate(vals):
    ctx = {"cycle": {}, "T_in": None, "m_gas": None, "econ": {}, "mult": {}, "extra": 0.0}
    reset()
    try:
        for p in PARAMS:
            if p["name"] in vals:
                p["fn"](vals[p["name"]], ctx)
        Pl = evaluate_plant(**CASE, T_in=ctx["T_in"], m_gas=ctx["m_gas"], **ctx["cycle"])
        (i0, tci0), (i1, _) = economics(Pl, ctx["econ"], ctx["mult"], [0.0, ctx["extra"]])
        r = {"ok": True, "net_el": Pl["net_el"] / 1e6, "NPV_model": i0["NPV"] / 1e6,
             "NPV_all": i1["NPV"] / 1e6, "LCOE": i0["LCOE"], "TCI": tci0 / 1e6,
             "T1": Pl["T1"] - K, "stack": Pl["hs"]["T_out"] - K,
             "HTR_dT": Pl["res"]["pinch"]["HTR"]["min_dT"],
             "LTR_dT": Pl["res"]["pinch"]["LTR"]["min_dT"], "error": ""}
        if not all(np.isfinite(r[k]) for k in ("net_el", "NPV_model", "NPV_all", "LCOE")):
            r.update(ok=False, error="non-finite result (heat exchanger cross or infinite cost)")
        return r
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:80]}"}
    finally:
        reset()


t0 = time.time()
base = evaluate({})
dt = time.time() - t0
print(f"Base case: net electrical power {base['net_el']:.4f} MW, NPV {base['NPV_all']:.3f} M$ "
      f"(earlier tables: 6.1971 MW, 7.981 M$)")
print(f"One evaluation takes {dt:.1f} s: tornado about {2 * len(PARAMS) * dt / 60:.0f} min, "
      f"Monte Carlo about {N_MC * dt / 60:.0f} min\n", flush=True)

# ======================================================================
# 1) One at a time
# ======================================================================
rows = []
for i, p in enumerate(PARAMS, 1):
    lo, hi = evaluate({p["name"]: p["low"]}), evaluate({p["name"]: p["high"]})
    ok = lo["ok"] and hi["ok"]
    nan = float("nan")
    rows.append({
        "Input": p["name"], "Group": p["group"], "Unit": p["unit"],
        "Low": p["low"], "Base": p["base"], "High": p["high"],
        "Net power at low (MW)": lo.get("net_el", nan) if lo["ok"] else nan,
        "Net power at high (MW)": hi.get("net_el", nan) if hi["ok"] else nan,
        "NPV at low (M$)": lo["NPV_all"] if lo["ok"] else nan,
        "NPV at high (M$)": hi["NPV_all"] if hi["ok"] else nan,
        "NPV swing (M$)": abs(hi["NPV_all"] - lo["NPV_all"]) if ok else nan,
        "Basis": p["basis"],
        "Note": "" if ok else f"infeasible: low {lo.get('error', 'ok')}; high {hi.get('error', 'ok')}"})
    if i % 8 == 0:
        print(f"  tornado: {i} of {len(PARAMS)} inputs done", flush=True)

tor = pd.DataFrame(rows).sort_values("NPV swing (M$)", ascending=False, na_position="last")
Path("tables").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)
Path("data/results").mkdir(parents=True, exist_ok=True)
out = tor.copy()
num = out.select_dtypes("number").columns
out[num] = out[num].round(4)
out.to_csv("tables/14_uncertainty_tornado.csv", index=False, encoding="utf-8-sig")
print("\nNPV swing (low to high of each input, others at base):")
print(out[["Input", "Low", "High", "NPV at low (M$)", "NPV at high (M$)",
           "NPV swing (M$)", "Net power at low (MW)", "Net power at high (MW)",
           "Note"]].head(16).to_string(index=False))

top = tor.dropna(subset=["NPV swing (M$)"]).head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(11, 7))
yy = np.arange(len(top))
b = base["NPV_all"]
ax.barh(yy, top["NPV at low (M$)"] - b, left=b, color="tab:blue", label="input at its low value")
ax.barh(yy, top["NPV at high (M$)"] - b, left=b, color="tab:orange", alpha=0.8,
        label="input at its high value")
ax.axvline(b, color="k", lw=1)
ax.axvline(0, color="red", lw=1, ls="--")
ax.set_yticks(yy)
ax.set_yticklabels([f"{r.Input}  [{r.Low:g} to {r.High:g} {r.Unit}]" for r in top.itertuples()],
                   fontsize=8)
ax.set_xlabel("NPV (M$)  (black line: base case; red dashed: NPV = 0)")
ax.set_title("Effect of each input on NPV, one at a time (design fixed)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, axis="x")
fig.tight_layout()
fig.savefig("figures/uncertainty_tornado.png", dpi=150)
plt.close(fig)

# ======================================================================
# 2) Monte Carlo, all inputs together
# ======================================================================
lows = np.array([p["low"] for p in PARAMS], float)
highs = np.array([p["high"] for p in PARAMS], float)
V = lows + qmc.LatinHypercube(d=len(PARAMS), seed=SEED).random(N_MC) * (highs - lows)
records = []
for k in range(N_MC):
    vals = {p["name"]: V[k, j] for j, p in enumerate(PARAMS)}
    r = evaluate(vals)
    records.append({**{p["name"]: V[k, j] for j, p in enumerate(PARAMS)}, **r})
    if (k + 1) % 25 == 0:
        print(f"  Monte Carlo: {k + 1} of {N_MC} samples", flush=True)

mc = pd.DataFrame(records)
good = mc[mc["ok"]].copy()
n_fail = len(mc) - len(good)
print(f"\nMonte Carlo: {len(good)} feasible samples, {n_fail} infeasible")
good.drop(columns=["ok", "error"]).to_csv("data/results/uncertainty_samples.csv", index=False)


def summary(s, label):
    q = s.quantile([0.05, 0.10, 0.50, 0.90, 0.95])
    return {"Quantity": label, "Mean": s.mean(), "P5": q[0.05], "P10": q[0.10],
            "P50": q[0.50], "P90": q[0.90], "P95": q[0.95], "Min": s.min(), "Max": s.max()}


stat = pd.DataFrame([
    summary(good["net_el"], "Net electrical power (MW)"),
    summary(good["NPV_model"], "NPV, modelled inputs only (M$)"),
    summary(good["NPV_all"], "NPV, incl. unmodelled capital (M$)"),
    summary(good["LCOE"], "LCOE, modelled inputs only (USD/MWh)"),
    summary(good["TCI"], "Total capital investment (M$)")]).round(3)
print(stat.to_string(index=False))
share = pd.DataFrame([
    ("Samples with NPV < 0, modelled inputs only", (good["NPV_model"] < 0).mean() * 100),
    ("Samples with NPV < 0, incl. unmodelled capital", (good["NPV_all"] < 0).mean() * 100),
    ("Samples where HTR or LTR minimum dT < 10 K (design constraint broken)",
     ((good["HTR_dT"] < 10) | (good["LTR_dT"] < 10)).mean() * 100),
    ("Infeasible samples (excluded above)", n_fail / len(mc) * 100)],
    columns=["Share", "Percent"]).round(1)
print(share.to_string(index=False))
stat.to_csv("tables/15_uncertainty_montecarlo.csv", index=False, encoding="utf-8-sig")
with open("tables/15_uncertainty_montecarlo.csv", "a", encoding="utf-8-sig") as f:
    f.write("\n")
    share.to_csv(f, index=False)

X = good[[p["name"] for p in PARAMS]]
drv = pd.DataFrame({"Rank correlation with NPV": X.corrwith(good["NPV_model"], method="spearman"),
                    "Rank correlation with net power": X.corrwith(good["net_el"], method="spearman")})
drv = drv.reindex(drv["Rank correlation with NPV"].abs().sort_values(ascending=False).index)
drv.round(3).to_csv("tables/16_uncertainty_drivers.csv", encoding="utf-8-sig")
print("\nStrongest drivers of NPV (Spearman rank correlation, modelled inputs only):")
print(drv.round(3).head(10).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
ax.hist(good["NPV_model"], bins=25, alpha=0.7, label="modelled inputs only")
ax.hist(good["NPV_all"], bins=25, alpha=0.6, label="incl. unmodelled capital")
ax.axvline(0, color="red", ls="--")
ax.axvline(base["NPV_model"], color="k", ls=":", label="base case")
ax.set_xlabel("NPV (M$)")
ax.set_ylabel("Samples")
ax.legend(fontsize=8)
ax = axes[1]
ax.hist(good["net_el"], bins=25, color="tab:green", alpha=0.7)
ax.axvline(base["net_el"], color="k", ls=":", label="base case")
ax.set_xlabel("Net electrical power (MW)")
ax.legend(fontsize=8)
for ax in axes:
    ax.grid(True)
fig.suptitle(f"Monte Carlo, {len(good)} samples (design fixed)")
fig.tight_layout()
fig.savefig("figures/uncertainty_montecarlo.png", dpi=150)
print("\nSaved: tables/14 to 16, figures/uncertainty_tornado.png, uncertainty_montecarlo.png, "
      "data/results/uncertainty_samples.csv")