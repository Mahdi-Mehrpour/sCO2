from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar, Bounds

from sco2.cycle_adv import recompression_adv, recompression_cascade
from sco2.heat_source_adv import EAF, GasStream, couple_heater, couple_cascade

K = 273.15
gas = GasStream(EAF["X"], EAF["p"], T0=298.15)
dTp, Tmin, mg = EAF["dT_pinch_heater"], EAF["T_stack_min"], EAF["m_gas"]


# ---------------- Baseline: single heater, x optimised for net power -------
def run_base(x):
    res = recompression_adv(x=x)
    hs = couple_heater(res, gas, EAF["T_in"], mg, dT_pinch=dTp, T_stack_min=Tmin)
    return res, hs


print("Optimising the baseline (x only)...", flush=True)
ob = minimize_scalar(lambda x: -run_base(x)[1]["m_co2"] * run_base(x)[0]["w_net"],
                     bounds=(0.05, 0.6), method="bounded", options={"xatol": 1e-3})
resb, hsb = run_base(ob.x)
Wb = hsb["m_co2"] * resb["w_net"]
print(f"  baseline: x = {ob.x:.3f}, net power = {Wb/1e6:.3f} MW\n", flush=True)


# ---------------- Cascade: x, y, T_b optimised for net power ---------------
# z = [x, y, T_b in (C/100)]
def run_cascade(z, full=False):
    res = recompression_cascade(x=z[0], y=z[1], T_b=z[2] * 100 + K, check_pinch=full)
    hs = couple_cascade(res, gas, EAF["T_in"], mg, dTp, dTp, Tmin)
    return res, hs


count = [0]


def objective(z):
    count[0] += 1
    if count[0] % 25 == 0:
        print(f"  evaluations: {count[0]}", flush=True)
    try:
        res, hs = run_cascade(z)
    except (ValueError, RuntimeError):
        return 0.0
    return -hs["m_co2"] * res["w_net"] / 1e6


print("Optimising the cascade (x, y, T_b)...", flush=True)
z0 = np.array([0.25, 0.15, 3.0])
simplex = np.array([z0, z0 + [0.08, 0, 0], z0 + [0, 0.08, 0], z0 + [0, 0, 0.8]])
opt = minimize(objective, z0, method="Nelder-Mead",
               bounds=Bounds([0.0, 0.02, 1.0], [0.6, 0.6, 5.0]),
               options={"initial_simplex": simplex, "maxiter": 120,
                        "xatol": 2e-3, "fatol": 1e-4})
print(f"  done after {count[0]} evaluations\n", flush=True)

res, hs = run_cascade(opt.x, full=True)
Wc = hs["m_co2"] * res["w_net"]
st = res["states"]
dest_b = hsb["ex_in"] - Wb - hsb["ex_out"]
dest_c = hs["ex_in"] - Wc - hs["ex_out"]
bal = res["q_in"] + res["q_b"] - res["w_net"] - res["q_out"]


def r(v, n=3):
    return round(v, n) if isinstance(v, (int, float, np.floating)) else v


rows = [
    ("Recompression fraction x", resb["x"], res["x"], "-"),
    ("Branch fraction y", "-", res["y"], "-"),
    ("Branch heater outlet temperature", "-", st[11]["T"] - K, "C"),
    ("CO2 mass flow (turbine)", hsb["m_co2"], hs["m_co2"], "kg/s"),
    ("Heat to the cycle, main heater", hsb["Q"] / 1e6, hs["Q1"] / 1e6, "MW"),
    ("Heat to the cycle, branch heater", "-", hs["Q2"] / 1e6, "MW"),
    ("Heat to the cycle, total", hsb["Q"] / 1e6, hs["Q"] / 1e6, "MW"),
    ("Gas temperature between the heaters", "-", hs["T_gas_mid"] - K, "C"),
    ("Gas stack temperature", hsb["T_out"] - K, hs["T_out"] - K, "C"),
    ("Share of available heat used", hsb["Q"] / hsb["Q_max"] * 100,
     hs["Q"] / hs["Q_max"] * 100, "%"),
    ("Net power", Wb / 1e6, Wc / 1e6, "MW"),
    ("Cycle thermal efficiency (on heat delivered)", resb["eta"] * 100, res["eta"] * 100, "%"),
    ("Heat recovery efficiency (net power / available heat)",
     Wb / hsb["Q_max"] * 100, Wc / hs["Q_max"] * 100, "%"),
    ("Exergy of gas at inlet", hsb["ex_in"] / 1e6, hs["ex_in"] / 1e6, "MW"),
    ("System exergy efficiency (net power / gas exergy)",
     Wb / hsb["ex_in"] * 100, Wc / hs["ex_in"] * 100, "%"),
    ("Exergy lost with the stack gas", hsb["ex_out"] / 1e6, hs["ex_out"] / 1e6, "MW"),
    ("Total exergy destruction (heaters, cycle, cooler)", dest_b / 1e6, dest_c / 1e6, "MW"),
    ("Main heater minimum temperature difference", hsb["min_dT"], hs["min_dT1"], "K"),
    ("Branch heater minimum temperature difference", "-", hs["min_dT2"], "K"),
    ("HTR minimum temperature difference", resb["pinch"]["HTR"]["min_dT"],
     res["pinch"]["HTR"]["min_dT"], "K"),
    ("LTR minimum temperature difference", resb["pinch"]["LTR"]["min_dT"],
     res["pinch"]["LTR"]["min_dT"], "K"),
    ("Limiting constraint of the gas coupling", "-", hs["binding"], ""),
    ("Cycle energy balance check", "-", bal, "J/kg"),
]
df = pd.DataFrame([{"Parameter": a, "Baseline (1 heater)": r(b),
                    "Cascade (2 heaters)": r(c), "Unit": u} for a, b, c, u in rows])
print(df.to_string(index=False))
print(f"\nNet power gain of the cascade: {(Wc/Wb-1)*100:+.1f} %")

Path("tables/adv").mkdir(parents=True, exist_ok=True)
df.to_csv("tables/adv/table6_cascade.csv", index=False, encoding="utf-8-sig")

# ---------------- Q-T diagram of both heaters -------------------------------
Q1, Q2 = hs["Q1"] / 1e6, hs["Q2"] / 1e6
f = hs["f"]
plt.figure(figsize=(8, 5.5))
plt.plot(f * Q2, hs["T_gas2"] - K, "r-", label="EAF gas")
plt.plot(Q2 + f * Q1, hs["T_gas1"] - K, "r-")
plt.plot(f * Q2, hs["T_co2_2"] - K, "b-", label="CO2, branch heater")
plt.plot(Q2 + f * Q1, hs["T_co2_1"] - K, "g-", label="CO2, main heater")
plt.xlabel("Heat transferred (MW)")
plt.ylabel("Temperature (°C)")
plt.title(f"Cascade heaters (min dT: branch {hs['min_dT2']:.1f} K, main {hs['min_dT1']:.1f} K)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("figures/adv/cascade_heaters_QT.png", dpi=150)
print("Figure saved: figures/adv/cascade_heaters_QT.png")
print("Table saved: tables/adv/table6_cascade.csv")