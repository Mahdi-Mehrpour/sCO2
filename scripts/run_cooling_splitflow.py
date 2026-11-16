import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sco2.cycle_adv import splitflow_adv
from sco2.heat_source_adv import EAF, GasStream, couple_cascade
from sco2.parasitics_splitflow import PARASITIC, parasitic_loads
from sco2.cooling_splitflow import (COOLING, compressor_inlet_T,
                                    cooler_performance)

K = 273.15
UA_MIN_COST = 8.6e5     # W/K, lower limit of the dry cooler cost correlation
                        # (economics file); below it the cost is extrapolated

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)
gas = GasStream(EAF["X"], EAF["p"], T0=298.15)

cases = [("Reference (T1 = 32 C fixed, dry air fans)", 305.15, "dry_air"),
         ("Dry air cooler", compressor_inlet_T("dry_air"), "dry_air"),
         ("Wet cooling tower", compressor_inlet_T("wet_tower"), "wet_tower")]

data, profiles = {}, {}
for label, T1, option in cases:
    res = splitflow_adv(T1=T1, y=opt["y"], T_b=opt["T_b"], pr=opt["pr"],
                        check_pinch=True)
    hs = couple_cascade(res, gas, EAF["T_in"], EAF["m_gas"],
                        EAF["dT_pinch_heater"], EAF["dT_pinch_heater"],
                        EAF["T_stack_min"])
    par = parasitic_loads(res, hs, gas, EAF["m_gas"], EAF["p"],
                          dict(PARASITIC, drive="motor"))
    cp = cooler_performance(option, res, hs)
    net_el = par["net_el"] + par["fan_air"] - cp["P_aux"]   # swap in this fan/pump
    q_heat = hs["m_co2"] * (res["q_in"] + res["q_b"])
    data[label] = {
        "Compressor inlet temperature T1 (C)": T1 - K,
        "Actual approach, CO2 outlet - coolant inlet (K)": cp["approach_actual"],
        "CO2 flow (kg/s)": hs["m_co2"],
        "Cooler duty (MW)": cp["Q"] / 1e6,
        "Cooler minimum temperature difference (K)": cp["min_dT"],
        "Cooler UA (kW/K)": cp["UA"] / 1e3,
        "Cooler fans / pumps (MW)": cp["P_aux"] / 1e6,
        cp["flow_name"]: cp["flow"],
        "Net shaft power (MW)": par["net_shaft"] / 1e6,
        "Net electrical power (MW)": net_el / 1e6,
        "Net electrical efficiency (%)": net_el / q_heat * 100,
        "Gas stack temperature (C)": hs["T_out"] - K,
        "HTR minimum temperature difference (K)": res["pinch"]["HTR"]["min_dT"],
        "LTR minimum temperature difference (K)": res["pinch"]["LTR"]["min_dT"],
        "Cooler UA below cost-correlation range": "yes" if cp["UA"] < UA_MIN_COST else "no",
    }
    profiles[label] = cp

df = pd.DataFrame(data)
ref = df.loc["Net electrical power (MW)"].iloc[0]
df.loc["Net electrical power vs reference (%)"] = [
    (v / ref - 1) * 100 for v in df.loc["Net electrical power (MW)"]]
df = df.fillna("-")
print(df.to_string(float_format=lambda v: f"{v:.3f}"))
print(f"\nT1 floor: {COOLING['T1_floor_C']} C (critical temperature + 1 K)")

Path("tables/splitflow").mkdir(parents=True, exist_ok=True)
df.to_csv("tables/splitflow/table7_cooling_options.csv", encoding="utf-8-sig")

Path("figures/splitflow").mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, (label, cp) in zip(axes, profiles.items()):
    ax.plot(cp["Q_axis"] / 1e6, cp["T_co2"] - K, "r-", label="CO2")
    ax.plot(cp["Q_axis"] / 1e6, cp["T_cool"] - K, "b-", label="Coolant")
    ax.set_title(f"{label}\nmin dT = {cp['min_dT']:.1f} K", fontsize=9)
    ax.set_xlabel("Heat rejected (MW), from the CO2 outlet end")
    ax.grid(True)
    ax.legend()
axes[0].set_ylabel("Temperature (°C)")
fig.tight_layout()
fig.savefig("figures/splitflow/cooler_QT.png", dpi=150)
print("Table saved: tables/splitflow/table7_cooling_options.csv")
print("Figure saved: figures/splitflow/cooler_QT.png")
