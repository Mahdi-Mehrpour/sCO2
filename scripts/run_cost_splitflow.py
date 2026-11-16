"""Economic optimisation of the split-flow cycle (no recompressor).
Maximises the annual net benefit (electricity revenue minus annualised
capital cost) for several pressure-cost exponents n_p.
All cost and economic values are placeholders (see cost_splitflow.py)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

from sco2.cycle_adv import splitflow_adv
from sco2.heat_source_adv import EAF, GasStream, couple_cascade
from sco2.cost_splitflow import capital_cost, economics

K = 273.15
DT_REC = 10.0                  # K, minimum temperature difference in HTR and LTR
PEN = 1.0                      # M$/yr per K of violation
NP_LIST = [0.0, 0.3, 0.6, 1.0] # pressure-cost exponents to test
NP_REF = 0.6                   # exponent used for the pressure sweep
LB = [0.02, 1.0, 1.5]          # y, T_b/100 (C), pr
UB = [0.60, 5.5, 4.5]
P1 = 7.7e6

gas = GasStream(EAF["X"], EAF["p"], T0=298.15)
dTp, Tmin, mg = EAF["dT_pinch_heater"], EAF["T_stack_min"], EAF["m_gas"]

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
z_maxpower = [opt["y"], (opt["T_b"] - K) / 100, opt["pr"]]
counter = [0]


def evaluate(z, n_p):
    res = splitflow_adv(y=z[0], T_b=z[1] * 100 + K, pr=z[2], check_pinch=True)
    hs = couple_cascade(res, gas, EAF["T_in"], mg, dTp, dTp, Tmin)
    c = capital_cost(res, hs, n_p=n_p)
    return res, hs, c, economics(c)


def violation(res):
    p = res["pinch"]
    return (max(0.0, DT_REC - p["HTR"]["min_dT"])
            + max(0.0, DT_REC - p["LTR"]["min_dT"]))


def obj(z, n_p):
    counter[0] += 1
    if counter[0] % 100 == 0:
        print(f"    evaluations: {counter[0]}", flush=True)
    try:
        res, hs, c, e = evaluate(z, n_p)
    except Exception:
        return 10.0
    if not np.isfinite(c["total"]):
        return 10.0
    return -e["benefit"] + PEN * violation(res)


def optimise(n_p, starts):
    lb, ub = np.array(LB, float), np.array(UB, float)
    out = []
    for z0 in starts:
        z0 = np.clip(np.array(z0, float), lb, ub)
        verts = [z0]
        for j, s in enumerate([0.05, 0.5, 0.3]):
            v = z0.copy()
            v[j] += s if z0[j] + s <= ub[j] else -s
            verts.append(v)
        o = minimize(obj, z0, args=(n_p,), method="Nelder-Mead",
                     bounds=Bounds(lb, ub),
                     options={"initial_simplex": np.array(verts), "maxfev": 250,
                              "xatol": 1e-3, "fatol": 1e-5})
        print(f"  start: benefit = {-o.fun:.4f} M$/yr, "
              f"z = {np.round(o.x, 3).tolist()}, evaluations = {o.nfev}", flush=True)
        out.append((o.fun, o.x))
    return min(out, key=lambda t: t[0])[1]


rows, breakdown, designs, prev = [], [], {}, None
for n_p in NP_LIST:
    print(f"\nn_p = {n_p}: maximising the annual net benefit...", flush=True)
    counter[0] = 0
    starts = [z_maxpower, prev if prev is not None else [0.25, 3.3, 3.0]]
    z = optimise(n_p, starts)
    prev = list(z)
    designs[n_p] = z
    res, hs, c, e = evaluate(z, n_p)
    resm, hsm, cm, em = evaluate(z_maxpower, n_p)       # max-power design
    flags = [n for n, v, lo, hi in zip(["y", "T_b", "pr"], z, LB, UB)
             if abs(v - lo) < 1e-3 or abs(v - hi) < 1e-3]
    rows.append({
        "n_p": n_p,
        "pr": z[2], "Max cycle pressure (MPa)": z[2] * P1 / 1e6,
        "y": z[0], "T_b (C)": z[1] * 100,
        "Net power (MW)": c["W_net_kW"] / 1e3,
        "Gas stack temperature (C)": hs["T_out"] - K,
        "Capital cost (M$)": c["total"] / 1e6,
        "Specific cost (USD/kW)": c["specific"],
        "Revenue (M$/yr)": e["revenue"],
        "Annualised capital (M$/yr)": e["annual_capital"],
        "Net benefit (M$/yr)": e["benefit"],
        "Simple payback (yr)": e["payback"],
        "Capital charge (USD/MWh)": e["lcoe_capital"],
        "HTR min dT (K)": res["pinch"]["HTR"]["min_dT"],
        "LTR min dT (K)": res["pinch"]["LTR"]["min_dT"],
        "Max-power design: net power (MW)": cm["W_net_kW"] / 1e3,
        "Max-power design: capital cost (M$)": cm["total"] / 1e6,
        "Max-power design: net benefit (M$/yr)": em["benefit"],
        "Benefit gain vs max-power design (M$/yr)": e["benefit"] - em["benefit"],
        "Variables at a bound": ", ".join(flags),
    })
    for name, v in c["items"].items():
        breakdown.append({"n_p": n_p, "Component": name, "Cost (M$)": v / 1e6,
                          "Share (%)": v / c["total"] * 100})

df = pd.DataFrame(rows).round(3)
dfb = pd.DataFrame(breakdown).round(3)
print("\n", df.T.to_string(header=False))
print("\nCost breakdown (M$):")
print(dfb.pivot(index="Component", columns="n_p", values="Cost (M$)").to_string())

# ---------------- Pressure sweep at NP_REF (y and T_b fixed) ----------------
zr = designs[NP_REF]
sweep = []
for pr_v in np.arange(2.0, 4.51, 0.25):
    try:
        res, hs, c, e = evaluate([zr[0], zr[1], pr_v], NP_REF)
        ok = violation(res) < 0.5
        sweep.append((pr_v, e["benefit"] if ok else np.nan,
                      c["W_net_kW"] / 1e3 if ok else np.nan,
                      c["total"] / 1e6 if ok else np.nan))
    except Exception:
        sweep.append((pr_v, np.nan, np.nan, np.nan))
sweep = np.array(sweep)

# ---------------- Save ------------------------------------------------------
T_OUT, F_OUT = Path("tables/cost"), Path("figures/cost")
T_OUT.mkdir(parents=True, exist_ok=True)
F_OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(T_OUT / "table11_cost_optimum.csv", index=False, encoding="utf-8-sig")
dfb.to_csv(T_OUT / "table12_cost_breakdown.csv", index=False, encoding="utf-8-sig")
with open("data/results/cost_optimum.json", "w") as f:
    json.dump({str(k): {"y": float(v[0]), "T_b": float(v[1] * 100 + K),
                        "pr": float(v[2])} for k, v in designs.items()}, f, indent=2)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
ax = axes[0]
bottom = np.zeros(len(NP_LIST))
for nm in dfb["Component"].unique():
    vals = np.array([dfb[(dfb.n_p == k) & (dfb.Component == nm)]["Cost (M$)"].iloc[0]
                     for k in NP_LIST])
    ax.bar([f"n_p = {k}" for k in NP_LIST], vals, bottom=bottom, label=nm)
    bottom += vals
ax.set_ylabel("Equipment cost (M$)")
ax.set_title("Cost of the economic optima")
ax.legend(fontsize=8)
ax = axes[1]
ax.plot(df["n_p"], df["Max cycle pressure (MPa)"], "o-")
ax.set_xlabel("Pressure-cost exponent n_p")
ax.set_ylabel("Optimal maximum pressure (MPa)")
ax.grid(True)
ax = axes[2]
ax.plot(sweep[:, 0], sweep[:, 1], "o-", color="tab:green")
ax.set_xlabel("Compressor pressure ratio")
ax.set_ylabel("Net benefit (M$/yr)", color="tab:green")
ax.grid(True)
ax2 = ax.twinx()
ax2.plot(sweep[:, 0], sweep[:, 2], "s--", color="tab:blue")
ax2.set_ylabel("Net power (MW)", color="tab:blue")
ax.set_title(f"Pressure sweep (n_p = {NP_REF}, y and T_b fixed)")
fig.tight_layout()
fig.savefig(F_OUT / "cost_results.png", dpi=150)
print(f"\nSaved: {T_OUT}/ and {F_OUT}/cost_results.png")