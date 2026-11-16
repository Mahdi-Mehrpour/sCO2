from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from sco2.cycle_adv import recompression_adv
from sco2.exergy_adv import exergy_analysis_adv
from sco2.heat_source_adv import EAF, GasStream, couple_heater

K = 273.15
res = recompression_adv()
ex = exergy_analysis_adv(res)
fe, dest = ex["flow_exergy"], ex["destruction"]

gas = GasStream(EAF["X"], EAF["p"], T0=298.15)
hs = couple_heater(res, gas, EAF["T_in"], EAF["m_gas"],
                   dT_pinch=EAF["dT_pinch_heater"], T_stack_min=EAF["T_stack_min"])

m = hs["m_co2"]
W = m * res["w_net"]                                   # W, net power
ex_gain = m * (fe[7] - fe[6])                          # exergy gained by the CO2
d_heater = hs["ex_in"] - hs["ex_out"] - ex_gain
d_others = m * sum(v for k, v in dest.items() if k not in ("Heater", "Cooler"))
d_cooler = m * dest["Cooler"]
bal = hs["ex_in"] - W - hs["ex_out"] - d_heater - d_others - d_cooler

rows = [
    ("Gas inlet temperature", EAF["T_in"] - K, "C"),
    ("Gas mass flow", EAF["m_gas"], "kg/s"),
    ("Gas molar mass", gas.M, "kg/kmol"),
    ("Gas outlet (stack) temperature", hs["T_out"] - K, "C"),
    ("Heat available down to minimum stack temperature", hs["Q_max"] / 1e6, "MW"),
    ("Heat delivered to the cycle", hs["Q"] / 1e6, "MW"),
    ("Share of available heat used", hs["Q"] / hs["Q_max"] * 100, "%"),
    ("CO2 mass flow in the cycle", m, "kg/s"),
    ("Heater minimum temperature difference", hs["min_dT"], "K"),
    ("Net power", W / 1e6, "MW"),
    ("Cycle thermal efficiency (on heat delivered)", res["eta"] * 100, "%"),
    ("Heat recovery efficiency (net power / available heat)", W / hs["Q_max"] * 100, "%"),
    ("Exergy of gas at inlet", hs["ex_in"] / 1e6, "MW"),
    ("Exergy efficiency of the whole system (net power / gas exergy)", W / hs["ex_in"] * 100, "%"),
    ("Exergy lost with the stack gas", hs["ex_out"] / 1e6, "MW"),
    ("Exergy destruction in the heater (gas to CO2)", d_heater / 1e6, "MW"),
    ("Exergy destruction in the other cycle components", d_others / 1e6, "MW"),
    ("Exergy loss in the cooler", d_cooler / 1e6, "MW"),
    ("Exergy balance check", bal / 1e6, "MW"),
]
df = pd.DataFrame([{"Parameter": a, "Value": round(b, 4), "Unit": c} for a, b, c in rows])
print(df.to_string(index=False))

Path("tables/adv").mkdir(parents=True, exist_ok=True)
df.to_csv("tables/adv/table5_heat_source.csv", index=False, encoding="utf-8-sig")

# Q-T diagram of the heater
Q = hs["f"] * res["q_in"] * m / 1e6
plt.figure(figsize=(7, 5))
plt.plot(Q, hs["T_gas"] - K, "r-", label="EAF exhaust gas")
plt.plot(Q, hs["T_co2"] - K, "b-", label="CO2 (heater)")
plt.xlabel("Heat transferred (MW)")
plt.ylabel("Temperature (°C)")
plt.title(f"Heater Q-T diagram (min dT = {hs['min_dT']:.1f} K)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("figures/adv/heater_QT.png", dpi=150)
print("\nFigure saved: figures/adv/heater_QT.png")
print("Table saved: tables/adv/table5_heat_source.csv")