import json
from dataclasses import replace
from pathlib import Path
import pandas as pd

from sco2.plant_splitflow import evaluate_plant
from sco2.economics_eaf import run, report, CEPCI
from sco2.economics_splitflow import build_inputs, fast_economics
from sco2.settings_splitflow import ECON_BASE

OUT = Path("tables")
OUT.mkdir(parents=True, exist_ok=True)

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
P = evaluate_plant(opt["y"], opt["T_b"], opt["pr"])
ci, e = build_inputs(P, ECON_BASE)
R = run(ci, e)
d, rows, ec, sp = R["size"], R["costs"]["rows"], R["econ"], R["speco"]
W_kWe = d["W_net"] * 1e3

print(f"Net power: economics module {d['W_net']:.4f} MW | plant model "
      f"{P['net_el'] / 1e6:.4f} MW")
print(f"SPECO balance residual: {sp['balance_residual']:.2e} USD/h")
for w in R["costs"]["warnings"]:
    print("WARNING:", w)

# ---------------- sizes ----------------------------------------------------
names = {"HTR": "HTR", "LTR": "LTR", "MH": "Main heater",
         "BH": "Branch heater", "cool": "Cooler"}
t_size = pd.DataFrame([{
    "Heat exchanger": n, "Duty (MW)": d[f"Q_{k}"],
    "End-point LMTD (K)": d[f"LMTD_ep_{k}"],
    "UA end-point (kW/K)": d[f"UA_ep_{k}"],
    "UA segmented (kW/K)": ci.UA_seg[k],
    "UA used for cost (kW/K)": d[f"UA_{k}"],
    "Segmented / end-point": ci.UA_seg[k] / d[f"UA_ep_{k}"]}
    for k, n in names.items()]).round(3)

# ---------------- component costs ------------------------------------------
Z = R["Z_dot"]
lst = [{"Component": n, "Purchased cost (k$)": r["PEC"] / 1e3,
        "Low (k$)": r["lo"] / 1e3, "High (k$)": r["hi"] / 1e3,
        "Cost rate (USD/h)": Z[n], "Source": r["source"]}
       for n, r in rows.items()]
lst.append({"Component": "Makeup water", "Purchased cost (k$)": 0.0,
            "Low (k$)": 0.0, "High (k$)": 0.0,
            "Cost rate (USD/h)": Z["_water_cost_rate"],
            "Source": f"{R['water_m3_yr']:,.0f} m3/yr"})
lst.append({"Component": "Total purchased equipment",
            "Purchased cost (k$)": R["PEC"] / 1e3, "Low (k$)": "", "High (k$)": "",
            "Cost rate (USD/h)": "", "Source": ""})
lst.append({"Component": "Total capital investment (with installation)",
            "Purchased cost (k$)": R["TCI"] / 1e3, "Low (k$)": ec["lo"]["TCI"] / 1e3,
            "High (k$)": ec["hi"]["TCI"] / 1e3, "Cost rate (USD/h)": "",
            "Source": f"{R['TCI'] / W_kWe:,.0f} USD/kWe net"})
t_cost = pd.DataFrame(lst)
num = t_cost.select_dtypes("number").columns
t_cost[num] = t_cost[num].round(2)

# ---------------- indicators -----------------------------------------------
t_ind = pd.DataFrame([
    ("Net electrical power (MW)", d["W_net"], "", ""),
    ("Total capital investment (M$)", R["TCI"] / 1e6, ec["lo"]["TCI"] / 1e6, ec["hi"]["TCI"] / 1e6),
    ("Capital per kWe (USD/kWe)", R["TCI"] / W_kWe, ec["lo"]["TCI"] / W_kWe, ec["hi"]["TCI"] / W_kWe),
    ("LCOE (USD/MWh)", ec["LCOE"], ec["lo"]["LCOE"], ec["hi"]["LCOE"]),
    ("NPV (M$)", ec["NPV"] / 1e6, ec["lo"]["NPV"] / 1e6, ec["hi"]["NPV"] / 1e6),
    ("IRR (%)", ec["IRR"] * 100, ec["lo"]["IRR"] * 100, ec["hi"]["IRR"] * 100),
    ("Simple payback (yr)", ec["PBP"], ec["lo"]["PBP"], ec["hi"]["PBP"]),
    ("Discounted payback (yr)", ec["DPBP"], ec["lo"]["DPBP"], ec["hi"]["DPBP"]),
    ("SPECO unit cost of electricity (USD/MWh)", sp["c_el_MWh"], "", ""),
], columns=["Indicator", "Base", "Low-capital case", "High-capital case"])
for c in t_ind.columns[1:]:
    t_ind[c] = t_ind[c].map(lambda v: round(v, 3) if isinstance(v, float) else v)

# ---------------- SPECO ----------------------------------------------------
t_speco = pd.DataFrame(sp["table"]).round(3)

# ---------------- sensitivities --------------------------------------------
extra_heater = (rows["Main heater"]["PEC"] + rows["Branch heater"]["PEC"]) / R["PEC"]


def scen(name, ci_=ci, e_=e, extra=0.0):
    o = fast_economics(ci_, e_, extra)
    return {"Scenario": name, "Capital (M$)": o["TCI"] / 1e6,
            "Capital (USD/kWe)": o["TCI"] / (o["W_net"] * 1e3),
            "LCOE (USD/MWh)": o["LCOE"], "NPV (M$)": o["NPV"] / 1e6,
            "IRR (%)": o["IRR"] * 100, "Simple payback (yr)": o["PBP"]}


t_sens = pd.DataFrame([
    scen("Base case"),
    scen("UA from end-point LMTD only", replace(ci, ua_basis="endpoint")),
    scen("Heater cost doubled", extra=extra_heater),
    scen("+30 % capital (ducts, heat buffer, engineering; placeholder)", extra=0.30),
    scen("Discount rate 5 %", e_=replace(e, r=0.05)),
    scen("Discount rate 10 %", e_=replace(e, r=0.10)),
    scen("Electricity value 0.06 USD/kWh", e_=replace(e, c_e=0.06)),
    scen("Pessimistic: 10 %, 0.06 USD/kWh, +30 % capital",
         e_=replace(e, r=0.10, c_e=0.06), extra=0.30),
]).round(2)

rates, prices = [0.05, 0.08, 0.10], [0.06, 0.08, 0.10]
t_grid = pd.DataFrame(
    [[fast_economics(ci, replace(e, r=r_, c_e=p_))["NPV"] / 1e6 for p_ in prices]
     for r_ in rates],
    index=[f"discount rate {r_ * 100:.0f} %" for r_ in rates],
    columns=[f"{p_:.2f} USD/kWh" for p_ in prices]).round(2)

# ---------------- assumptions (read from the code) --------------------------
desc = {
    "cepci_target": ("Cost year index (CEPCI of the cost year)", "-"),
    "f_inst": ("Installation and auxiliaries, fraction of purchased cost", "-"),
    "r": ("Discount rate", "1/yr"),
    "N": ("Economic life", "yr"),
    "hours": ("Operating hours (calendar days x 24 h x availability)", "h/yr"),
    "c_e": ("Value of the electricity (avoided purchase price, site assumption)", "USD/kWh"),
    "c_OM": ("O&M cost", "USD/kWe-yr"),
    "er": ("Escalation of O&M and water price", "1/yr"),
    "dr": ("Output degradation", "1/yr"),
    "salvage": ("Salvage value, fraction of total capital investment", "-"),
    "water_price": ("Makeup water price (site assumption)", "USD/m3"),
    "c_gas": ("Cost of the waste-heat exergy (treated as free)", "USD/GJ"),
}
A = [{"Category": "Economics", "Assumption": text, "Value": getattr(e, k), "Unit": unit}
     for k, (text, unit) in desc.items()]
A += [{"Category": "Sizing", "Assumption": t, "Value": v, "Unit": u} for t, v, u in [
    ("Heat exchanger UA used for cost", ci.ua_basis + " (larger of end-point and segmented UA)", ""),
    ("Segmented UA: 30 segments for recuperators, 20 for heaters and cooler", "", ""),
    ("Cost year", f"CEPCI {CEPCI[2023]} (2023)", ""),
    ("Currency", "USD (nominal values of the cost references; no conversion)", "")]]
A += [{"Category": "Cost data", "Assumption": f"{n}: {r['source']}", "Value":
       f"band {r['lo'] / r['PEC'] - 1:+.0%} / {r['hi'] / r['PEC'] - 1:+.0%}", "Unit": ""}
      for n, r in rows.items()]
A += [{"Category": "Cost data", "Assumption": t, "Value": "", "Unit": ""} for t in [
    "Gas heater cost band widened upward (dust, fouling, refractory); not validated for EAF gas",
    "Cooling tower cost: modified Zanker formula reconstructed from the reference text; band assumed",
    "Weiland correlations used inside or near their size ranges; any extrapolation is listed as a warning"]]
A += [{"Category": "Not included", "Assumption": t, "Value": "", "Unit": ""} for t in [
    "Gas ducts, dust handling and filters, controls, grid connection, engineering and contingency",
    "Heat buffer (thermal storage) between the gas and the cycle",
    "Pressure effect on cost (the correlations have no pressure factor)",
    "Taxes, financing structure, incentives"]]
A += [{"Category": "SPECO", "Assumption": t, "Value": "", "Unit": ""} for t in [
    "Physical exergy is not split into thermal and mechanical parts",
    "Cooler and cooling tower are dissipative: their cost, tower power and water are charged to the product",
    "O&M is shared in proportion to each component's purchased cost"]]
t_ass = pd.DataFrame(A)

# ---------------- save and print -------------------------------------------
NUM = {"costs": "06", "indicators": "07", "sensitivity": "08",
       "npv_grid": "09", "speco": "10", "sizes": "11"}
tables = {"sizes": t_size, "costs": t_cost, "indicators": t_ind, "speco": t_speco,
          "sensitivity": t_sens, "npv_grid": t_grid}
for name, df in tables.items():
    df.to_csv(OUT / f"{NUM[name]}_economics_{name}.csv", encoding="utf-8-sig",
              index=name == "npv_grid")
(OUT / "economics_report.md").write_text(report(R), encoding="utf-8")

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 70)
for name in ("sizes", "costs", "indicators", "sensitivity", "npv_grid"):
    print(f"\n=== {name} ===")
    print(tables[name].to_string(index=name == "npv_grid"))
print(f"\nSaved in {OUT}/")