"""Checks of the off-design water-cooled cooler (straight-channel PCHE + wet cooling tower) against the design point.

A  Design: the design water inlet temperature and flows must give the design duty, T1 = 32 C, water outlet and minimum dT.
B  Wet-bulb temperature: water and tower air flow fixed at design; T1 is the result (cold water from the tower model).
C  Flow: water flow (and tower air flow at constant L/G, or water flow alone) at the design wet bulb: how much T1 falls with
   more flow, what the pump and the tower fan cost. The water side is laminar at design, so the gain is small.
D  Control: the flow (water and tower air at constant L/G) that holds T1 = 32 C at each wet-bulb temperature, up to the pump limit
   (1.3 x design) and the fan limit; beyond that T1 rises. Auxiliary power and evaporation are reported.
E  Water-side Reynolds number per segment (laminar below 2300) at design and at the highest flow.
F  Design temperature profile of the cooler with the minimum dT and the cp peak (pseudo-critical region).
The CO2 inlet enthalpy, flow and outlet pressure are the design values in every case; in the cycle they move as well.
Output: tables/water_cooler_checks.csv, figures/water_cooler_checks.png (the figure is refreshed on every run).
Run from the repository folder:  python "Simple Recompression - Off-Design/scripts/Cooler/check_water_cooler.py"
The first run also builds a CO2 property table (cached afterwards in tables/co2_cooler_table.npz). Run cooler_resistance_shares.py first."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # scripts folder: design_point
sys.path.insert(0, str(HERE))            # this folder first: water_cooler, cooling_tower

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import design_point as dp
from cooling_tower import CoolingTower
from water_cooler import WaterCooler, K, load_shares

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": False})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b7280"
T1_TARGET = dp.T1

t0 = time.time()
ref = pd.read_csv(HERE / "cooler_reference_data.csv").set_index("parameter")["value"].astype(float)
shares_file = dp.TABLES / "cooler_resistance_shares.csv"
if not shares_file.exists():
    sys.exit(f"{shares_file} is missing: run cooler_resistance_shares.py first.")
res, hs, x = dp.load_design()
D = dp.water_cooler_design(res, hs, x)
shares = load_shares(shares_file)
n = len(shares)
T_w_in_ref = ref["T_wb_des"] + ref["approach"] + ref["dT_loop"]
m_w_ref = None
cooler = WaterCooler(shares, D, ref, table_path=dp.TABLES / "co2_cooler_table.npz")                      # tower is attached after its design values are known
tower = CoolingTower(ref["T_wb_des"], cooler.T_w_in_des - K - ref["dT_loop"], cooler.T_w_out_des - K - ref["dT_loop"],
                     cooler.m_w_des, ref["LG_des"], ref["n_merkel"])
cooler.tower = tower
h_in, p_out, m_c = D["co2_in"]["h"], D["co2_out"]["p"], D["m_co2"]
WB = ref["T_wb_des"]
rows = []
print(f"CO2 property table ready in {time.time() - t0:.0f} s (built on the first run, then read from tables/co2_cooler_table.npz)")
acc = cooler.tab.accuracy()
print("Table vs CoolProp (7.6-9.5 MPa), largest deviation: T {T:.4f} K, rho {D:.2e}, mu {V:.2e} (relative); k and cp come from CoolProp".format(**acc))


def add(check, **kw):
    rows.append({"check": check, **kw})


print(f"Design: duty {D['Q'] / 1e6:.3f} MW, CO2 {m_c:.2f} kg/s, water {cooler.m_w_des:.1f} kg/s "
      f"({cooler.T_w_in_des - K:.2f} -> {cooler.T_w_out_des - K:.2f} C), UA {cooler.UA_total_des / 1e6:.3f} MW/K, {n} segments")
print(f"Tower: air {cooler.m_air_des:.1f} kg/s, fan {cooler.P_fan_des / 1e3:.1f} kW; pump {cooler.P_pump_des / 1e3:.1f} kW "
      f"(cooler water dp {cooler.dp_w_des.sum() / 1e3:.1f} kPa + loop {cooler.loop_dp / 1e3:.0f} kPa)")

# ---- A design
rA = cooler.solve(cooler.T_w_in_des, cooler.m_w_des, h_in, p_out, m_c)
rAc = cooler.solve_coupled(WB, cooler.m_w_des, cooler.m_air_des, h_in, p_out, m_c)
print("\nA  Design reproduction")
print(f"   given water inlet : Q {rA['Q'] / 1e6:.4f} MW (design {D['Q'] / 1e6:.4f}), T1 {rA['T1'] - K:.3f} C (design {D['T1'] - K:.3f}), "
      f"water out {rA['T_w_out'] - K:.3f} C (design {cooler.T_w_out_des - K:.3f}), min dT {rA['min_dT']:.2f} K, "
      f"UA {rA['UA_total'] / 1e6:.4f} MW/K")
print(f"   with the tower    : T1 {rAc['T1'] - K:.3f} C, cold water {rAc['T_w_in'] - K:.2f} C, approach {rAc['approach']:.2f} K, "
      f"dp CO2 {rAc['dp_co2'] / 1e3:.1f} kPa (design {(D['co2_in']['p'] - p_out) / 1e3:.1f}), water {rAc['dp_water'] / 1e3:.1f} kPa")
add("A", T_wb=WB, m_w=cooler.m_w_des, m_air=cooler.m_air_des, Q_MW=rAc["Q"] / 1e6, T1_C=rAc["T1"] - K, T_w_in_C=rAc["T_w_in"] - K,
    T_w_out_C=rAc["T_w_out"] - K, min_dT=rAc["min_dT"], P_aux_kW=rAc["P_aux"] / 1e3)

# ---- B wet bulb
print("\nB  Wet-bulb temperature, design water and air flow")
print(f"   {'T_wb':>6} {'cold water':>11} {'approach':>9} {'T1':>8} {'Q':>8} {'min dT':>7}")
B = []
T0 = None
for Twb in (10.0, 15.0, 20.0, 25.0, 28.0, 30.0):
    r = cooler.solve_coupled(Twb, cooler.m_w_des, cooler.m_air_des, h_in, p_out, m_c, T_w_out0=T0)
    T0 = r["T_w_out"]
    B.append((Twb, r))
    print(f"   {Twb:6.1f} {r['T_w_in'] - K:11.2f} {r['approach']:9.2f} {r['T1'] - K:8.2f} {r['Q'] / 1e6:8.3f} {r['min_dT']:7.2f}")
    add("B", T_wb=Twb, T_w_in_C=r["T_w_in"] - K, approach=r["approach"], T1_C=r["T1"] - K, Q_MW=r["Q"] / 1e6, min_dT=r["min_dT"],
        T_w_out_C=r["T_w_out"] - K)
sl = (B[-1][1]["T1"] - B[0][1]["T1"]) / (B[-1][0] - B[0][0])
print(f"   mean slope dT1/dT_wb = {sl:.2f} K/K")

# ---- C flow
print("\nC  Flow at the design wet bulb (relative to design)")
print(f"   {'flow':>5} | {'water only: T1':>14} {'UA MW/K':>8} | {'L/G const: T1':>13} {'cold water':>10} {'pump kW':>8} {'fan kW':>7}")
C = []
for fr in (0.6, 0.8, 1.0, 1.15, 1.3):
    rw = cooler.solve_coupled(WB, fr * cooler.m_w_des, cooler.m_air_des, h_in, p_out, m_c)
    rb = cooler.solve_coupled(WB, fr * cooler.m_w_des, fr * cooler.m_air_des, h_in, p_out, m_c)
    C.append((fr, rw, rb))
    print(f"   {fr:5.2f} | {rw['T1'] - K:14.2f} {rw['UA_total'] / 1e6:8.3f} | {rb['T1'] - K:13.2f} {rb['T_w_in'] - K:10.2f} "
          f"{rb['P_pump'] / 1e3:8.1f} {rb['P_fan_tower'] / 1e3:7.1f}")
    add("C", flow_rel=fr, T1_water_only_C=rw["T1"] - K, UA_water_only_MW_K=rw["UA_total"] / 1e6, T1_const_LG_C=rb["T1"] - K,
        T_w_in_const_LG_C=rb["T_w_in"] - K, P_pump_kW=rb["P_pump"] / 1e3, P_fan_kW=rb["P_fan_tower"] / 1e3, min_dT=rb["min_dT"])

# ---- D control
print(f"\nD  Control: flow (water and tower air, constant L/G) to hold T1 = {T1_TARGET - K:.0f} C; limit {cooler.m_w_max / cooler.m_w_des:.2f} x")
print(f"   {'T_wb':>6} {'flow':>6} {'T1':>7} {'cold water':>11} {'min dT':>7} {'P_aux kW':>9} {'evap kg/s':>10}  note")
Dres = []
r_prev = 0.7
for Twb in (10.0, 15.0, 20.0, 25.0, 28.0):
    r, limited = cooler.flow_for_T1(T1_TARGET, Twb, h_in, p_out, m_c, r0=r_prev)
    r_prev = r["m_w"] / cooler.m_w_des
    Dres.append((Twb, r, limited))
    note = "flow limit reached, T1 rises" if limited else ("" if r["min_dT"] > 0.5 else "pinch below 0.5 K")
    if r["cross"]:
        note += " temperature cross"
    print(f"   {Twb:6.1f} {r['m_w'] / cooler.m_w_des:6.2f} {r['T1'] - K:7.2f} {r['T_w_in'] - K:11.2f} {r['min_dT']:7.2f} "
          f"{r['P_aux'] / 1e3:9.1f} {r['m_evap']:10.2f}  {note}")
    add("D", T_wb=Twb, flow_rel=r["m_w"] / cooler.m_w_des, limited=limited, T1_C=r["T1"] - K, T_w_in_C=r["T_w_in"] - K,
        min_dT=r["min_dT"], P_pump_kW=r["P_pump"] / 1e3, P_fan_kW=r["P_fan_tower"] / 1e3, m_evap=r["m_evap"], Q_MW=r["Q"] / 1e6)

# ---- E Reynolds number
rE_hi = C[-1][2]
print("\nE  Water-side Reynolds number: segments above 2300 "
      f"{rA['n_turb']}/{n} at design (Re max {rA['Re_w_max']:.0f}), {rE_hi['n_turb']}/{n} at 1.3 x flow (Re max {rE_hi['Re_w_max']:.0f})")
print(f"   cp peak at {rA['f_pc'] * 100:.0f} % of the exchanger from the CO2 outlet")
for i in range(n):
    add("E", seg=i + 1, Re_water_design=rA["Re_w"][i], Re_water_1p3=rE_hi["Re_w"][i])
print(f"\nRun time {time.time() - t0:.0f} s")

pd.DataFrame(rows).round(5).to_csv(dp.TABLES / "water_cooler_checks.csv", index=False, encoding="utf-8-sig")

# ---------------- figure
fig, ax = plt.subplots(2, 3, figsize=(14, 8))
a = ax[0, 0]
a.plot([b[0] for b in B], [b[1]["T1"] - K for b in B], "o-", color=BLUE, label="$T_1$ (design flows)")
a.plot([b[0] for b in B], [b[1]["T_w_in"] - K for b in B], "s--", color=AQUA, label="cold water from the tower")
a.plot([b[0] for b in B], [b[0] for b in B], ":", color=GREY, label="wet bulb")
a.axhline(T1_TARGET - K, color=GREY, lw=0.8)
a.set(xlabel="Wet-bulb temperature (°C)", ylabel="Temperature (°C)", title="B  Wet bulb, fixed flows")
a.legend(fontsize=8)
a = ax[0, 1]
fr = [c[0] for c in C]
a.plot(fr, [c[1]["T1"] - K for c in C], "o-", color=ORANGE, label="water flow only (air fixed)")
a.plot(fr, [c[2]["T1"] - K for c in C], "s-", color=BLUE, label="water and air together (L/G const)")
a.set(xlabel="Flow relative to design", ylabel="$T_1$ (°C)", title="C  Effect of flow at design wet bulb")
a.legend(fontsize=8)
a = ax[0, 2]
a.plot([d[0] for d in Dres], [d[1]["m_w"] / cooler.m_w_des for d in Dres], "o-", color=BLUE, label="flow needed")
a.axhline(cooler.m_w_max / cooler.m_w_des, color=ORANGE, ls="--", label="pump limit")
a.set(xlabel="Wet-bulb temperature (°C)", ylabel="Flow relative to design", title="D  Flow needed to hold T1 = 32 °C")
a2 = a.twinx()
a2.plot([d[0] for d in Dres], [d[1]["T1"] - K for d in Dres], "s:", color=AQUA, label="$T_1$ reached")
a2.set_ylabel("$T_1$ (°C)")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
a = ax[1, 0]
a.bar(np.arange(len(Dres)) - 0.2, [d[1]["P_pump"] / 1e3 for d in Dres], 0.4, color=BLUE, label="pump")
a.bar(np.arange(len(Dres)) + 0.2, [d[1]["P_fan_tower"] / 1e3 for d in Dres], 0.4, color=ORANGE, label="tower fan")
a.set_xticks(np.arange(len(Dres)), [f"{d[0]:.0f}" for d in Dres])
a.set(xlabel="Wet-bulb temperature (°C)", ylabel="Auxiliary power (kW)", title="D  Auxiliaries at T1 control")
a.legend(fontsize=8)
a = ax[1, 1]
seg = np.arange(1, n + 1)
a.plot(seg, rA["Re_w"], "-", color=BLUE, label="design")
a.plot(seg, rE_hi["Re_w"], "-", color=ORANGE, label="1.3 x flow")
a.axhline(2300, color=GREY, ls="--", lw=0.8)
a.set(xlabel="Segment (1 = CO2 outlet / water inlet)", ylabel="Water-side Re", title="E  Water-side Reynolds number")
a.legend(fontsize=8)
a = ax[1, 2]
f = np.linspace(0, 1, n + 1)
a.plot(f, rA["T_co2"] - K, color=ORANGE, label="CO2")
a.plot(f, rA["T_water"] - K, color=BLUE, label="water")
a.axvline(rA["f_pc"], color=GREY, ls=":", lw=1, label="cp peak segment")
a.set(xlabel="Fraction of the duty from the CO2 outlet", ylabel="Temperature (°C)",
      title=f"F  Design profile, min dT {rA['min_dT']:.1f} K")
a.legend(fontsize=8)
for q in ax.ravel():
    q.grid(alpha=0.25)
fig.suptitle("Off-design water-cooled cooler (PCHE) with a wet cooling tower, design wet bulb "
             f"{WB:.0f} °C", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(dp.FIGURES / "water_cooler_checks.png", dpi=170)
print(f"Saved: {dp.TABLES / 'water_cooler_checks.csv'} and {dp.FIGURES / 'water_cooler_checks.png'}")
