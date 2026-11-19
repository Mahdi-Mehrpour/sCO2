"""Why is the stack at 222 C and not 150 C? (685 C, 48.4 kg/s design)

Prints (1) the cycle states, (2) the gas and CO2 heat-capacity rates in the
branch heater, (3) a sweep of the branch fraction y and the branch outlet
temperature T_b: stack temperature, net power and the limiting constraint.

Run: python scripts/diag_stack_ref.py"""
import json
import pandas as pd

from sco2.plant_splitflow import evaluate_plant, get_gas

K = 273.15
with open("data/results/splitflow_optimum_ref.json") as f:
    o = json.load(f)
gas = get_gas()
T_in, mg = o["T_in_K"], o["m_gas"]
P = evaluate_plant(o["y"], o["T_b"], o["pr"], T_in=T_in, m_gas=mg)
res, hs = P["res"], P["hs"]
st = res["states"]

print("Cycle states (design):")
for k in sorted(st):
    s = st[k]
    print(f"  state {k:>2}: T = {s['T'] - K:7.1f} C, p = {s['p'] / 1e6:6.2f} MPa, "
          f"h = {s['h'] / 1e3:8.1f} kJ/kg")
y, m = res["y"], hs["m_co2"]
print(f"\nCO2 flow {m:.2f} kg/s, branch fraction y = {y:.4f} -> branch flow {y * m:.2f} kg/s")

h = lambda T: float(gas.h(T))
Tmid, Tout = hs.get("T_gas_mid"), hs["T_out"]
print(f"Gas: in {T_in - K:.1f} C -> after the main heater {Tmid - K:.1f} C -> stack {Tout - K:.1f} C "
      f"(limiting: {hs['binding']})")
Q_bh = y * m * (st[11]["h"] - st[2]["h"])
C_co2 = Q_bh / (st[11]["T"] - st[2]["T"])
C_gas = mg * (h(Tmid) - h(Tout)) / (Tmid - Tout)
print(f"Branch heater: CO2 {st[2]['T'] - K:.0f} -> {st[11]['T'] - K:.0f} C, duty {Q_bh / 1e6:.2f} MW")
print(f"  heat-capacity rate: CO2 branch {C_co2 / 1e3:.1f} kW/K | gas {C_gas / 1e3:.1f} kW/K "
      f"(ratio {C_co2 / C_gas:.2f})")
print(f"  gas cooling in the branch heater {Tmid - Tout:.0f} K vs CO2 heating "
      f"{st[11]['T'] - st[2]['T']:.0f} K")
Q_150 = mg * (h(Tout) - h(423.15))
print(f"Heat in the gas between the stack temperature and 150 C: {Q_150 / 1e6:.2f} MW "
      f"(heat available above 150 C: {hs['Q_max'] / 1e6:.2f} MW)\n")

rows = []
for yy in (0.20, o["y"], 0.35, 0.45, 0.55):
    for Tb in (250, 300, 350, o["T_b"] - K, 420):
        try:
            Pl = evaluate_plant(yy, Tb + K, o["pr"], T_in=T_in, m_gas=mg)
            rows.append({"y": round(yy, 3), "T_b (C)": round(Tb, 1),
                         "Stack (C)": round(Pl["hs"]["T_out"] - K, 1),
                         "Net power (MW)": round(Pl["net_el"] / 1e6, 3),
                         "Cycle eff. (%)": round(Pl["res"]["eta"] * 100, 1),
                         "Limiting": Pl["hs"]["binding"]})
        except Exception as e:
            rows.append({"y": round(yy, 3), "T_b (C)": round(Tb, 1), "Stack (C)": None,
                         "Net power (MW)": None, "Cycle eff. (%)": None,
                         "Limiting": f"infeasible ({type(e).__name__})"})
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("Sweep of y and T_b (pr fixed, same gas):")
print(df.to_string(index=False))
df.to_csv("tables/29_stack_diagnostic_ref.csv", index=False, encoding="utf-8-sig")
print("\nSaved: tables/29_stack_diagnostic_ref.csv")
