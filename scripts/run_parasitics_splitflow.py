import json
from pathlib import Path
import pandas as pd

from sco2.cycle_adv import splitflow_adv
from sco2.heat_source_adv import EAF, GasStream, couple_cascade
from sco2.parasitics_splitflow import PARASITIC, parasitic_loads

with open("data/results/splitflow_optimum.json") as f:
    opt = json.load(f)

gas = GasStream(EAF["X"], EAF["p"], T0=298.15)
res = splitflow_adv(y=opt["y"], T_b=opt["T_b"], pr=opt["pr"], check_pinch=False)
hs = couple_cascade(res, gas, EAF["T_in"], EAF["m_gas"], EAF["dT_pinch_heater"],
                    EAF["dT_pinch_heater"], EAF["T_stack_min"])

out = {}
for drive in ("motor", "shaft"):
    par = dict(PARASITIC, drive=drive)
    out[drive] = parasitic_loads(res, hs, gas, EAF["m_gas"], EAF["p"], par)

rows = [("Turbine shaft power", "W_turbine"),
        ("Compressor shaft power", "W_compressor"),
        ("Net shaft power (cycle model)", "net_shaft"),
        ("Gearbox losses", "loss_gearbox"),
        ("Generator losses", "loss_generator"),
        ("Compressor motor losses", "loss_motor"),
        ("Cooler fans", "fan_air"),
        ("Off-gas fan", "fan_gas"),
        ("Net electrical power", "net_el")]
table = [{"Item": name,
          "Motor-driven compressor (MW)": round(out["motor"][k] / 1e6, 4),
          "Shaft-driven compressor (MW)": round(out["shaft"][k] / 1e6, 4)}
         for name, k in rows]
for name, f_ in [("Net electrical / net shaft power (%)", lambda o: o["net_el"] / o["net_shaft"] * 100),
                 ("Net electrical efficiency (%)", lambda o: o["eta_net_el"] * 100),
                 ("Closure check (MW, should be 0)", lambda o: o["closure"] / 1e6)]:
    table.append({"Item": name,
                  "Motor-driven compressor (MW)": round(f_(out["motor"]), 4),
                  "Shaft-driven compressor (MW)": round(f_(out["shaft"]), 4)})
df = pd.DataFrame(table)
print(f"CO2 flow {hs['m_co2']:.2f} kg/s, cooler duty {hs['m_co2'] * res['q_out'] / 1e6:.2f} MW")
print(f"Cooler air flow {out['motor']['V_air']:.0f} m3/s, off-gas flow {out['motor']['V_gas']:.1f} m3/s\n")
print(df.to_string(index=False))

Path("tables/splitflow").mkdir(parents=True, exist_ok=True)
df.to_csv("tables/splitflow/table6_parasitics.csv", index=False, encoding="utf-8-sig")
print("\nTable saved: tables/splitflow/table6_parasitics.csv")