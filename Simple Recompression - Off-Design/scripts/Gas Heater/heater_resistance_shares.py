"""Gas heater (EAF off-gas -> sCO2), finned-tube bundle with sCO2 inside the tubes: thermal-resistance shares.

For every one of the 20 equal-duty design segments of the cycle model, the script splits the design resistance into
gas side, CO2 side, wall and fouling. The off-design heater model scales only the flow-dependent parts with these shares.
The design point is read from the cycle model (design_point.load_design), so the segment temperatures, duties, flows,
pressures and the gas composition are exactly those of the design study; nothing is typed in here.

Correlations
  Gas side : Briggs & Young (1963), finned tubes, staggered bank:  Nu = 0.134 Re^0.681 Pr^(1/3) (s/hf)^0.2 (s/tf)^0.1134
             (Re on the outer tube diameter and the velocity in the minimum free area; valid for Re about 1100-18000)
  Fin      : Schmidt approximation for annular fins, overall surface efficiency eta_o
  CO2 side : Gnielinski with the Petukhov friction factor
  Wall     : cylindrical conduction
  Fouling  : gas-side R_f'' on the finned surface (input, independent of flow)
Gas properties: ideal-gas mixture of pure CoolProp fluids at 1 atm (Wilke mixing for viscosity and conductivity), used only
for transport properties; the gas temperatures themselves come from the cycle model. CO2 properties: CoolProp at the local h, p.
The geometry below is a REFERENCE design (assumptions): change it to your own.

Output: tables/heater_resistance_shares[_<tag>].csv (one row per segment) and a summary on screen.
Run from the repository folder:
  python "Simple Recompression - Off-Design/scripts/Gas Heater/heater_resistance_shares.py"
  python "Simple Recompression - Off-Design/scripts/Gas Heater/heater_resistance_shares.py" --rf 0.0015 --tag fouled"""
import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                  # scripts/: design_point.py

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

import design_point as dp
from heater import GasTransport

ap = argparse.ArgumentParser()
ap.add_argument("--rf", type=float, default=0.0, help="gas-side fouling resistance R_f'' (m2.K/W), e.g. 0.0015 for dusty gas")
ap.add_argument("--tag", default="", help="suffix of the output file")
args = ap.parse_args()

# ---------------- reference geometry (ASSUMPTIONS) from heater_reference_data.csv ----------------
ref = pd.read_csv(HERE / "heater_reference_data.csv").set_index("parameter")["value"].astype(float)
Do, Di, hf, tf, nf = ref["Do"], ref["Di"], ref["hf"], ref["tf"], ref["nf"]
St, Sl, k_fin, k_wall = ref["St"], ref["Sl"], ref["k_fin"], ref["k_wall"]
Vmax_design, G_c_target, L_tube, eps_rough = ref["Vmax_design"], ref["G_c_target"], ref["L_tube"], ref["eps_rough"]
Rf_gas = args.rf                                      # m2.K/W gas-side fouling on the finned surface (command line)

# ---------------- design point from the cycle model ----------------
res, hs, x = dp.load_design()
st6, st7 = res["states"][6], res["states"][7]
m_c, m_g = hs["m_co2"], dp.EAF["m_gas"]
Q = hs["Q"]
f_seg = np.asarray(hs["f"], float)                    # 0 = CO2 inlet / gas outlet, 1 = CO2 outlet / gas inlet
N_SEG = len(f_seg) - 1
T_gas = np.asarray(hs["T_gas"], float)                # K at the segment boundaries (cycle model, Cantera gas table)
T_co2 = np.asarray(hs["T_co2"], float)
h_c_b = st6["h"] + f_seg * Q / m_c                    # J/kg of CO2 at the boundaries
p_c_b = st6["p"] + f_seg * (st7["p"] - st6["p"])      # Pa, linear as in the cycle model
# gas transport properties (CO lumped into N2 inside GasTransport; same object the off-design heater model uses)
TR = GasTransport(dp.EAF["X"])
gas_props = TR.props


# ---------------- geometry per metre of tube ----------------
Df = Do + 2 * hf
s_fin = 1.0 / nf - tf
A_fin = nf * (2 * math.pi / 4 * (Df ** 2 - Do ** 2) + math.pi * Df * tf)
A_base = math.pi * Do * (1 - nf * tf)
A_o, A_i = A_fin + A_base, math.pi * Di
block = Do + 2 * hf * tf * nf
sigma = (St - block) / St
R_wall = math.log(Do / Di) / (2 * math.pi * k_wall)


def fin_eff(h):
    m = math.sqrt(2 * h / (k_fin * tf)); ro = Do / 2; ratio = Df / Do
    z = m * ro * (ratio - 1) * (1 + 0.35 * math.log(ratio))
    return 1 - (A_fin / A_o) * (1 - math.tanh(z) / z)


T_gas_mean = 0.5 * (T_gas[0] + T_gas[-1])
A_min = m_g / (gas_props(T_gas_mean)["rho"] * Vmax_design)
N_t_row = A_min / sigma / (St * L_tube)
n_par = m_c / (G_c_target * math.pi * Di ** 2 / 4)
G_c = m_c / (n_par * math.pi * Di ** 2 / 4)

print(f"Design: gas {T_gas[-1] - dp.K:.2f} -> {T_gas[0] - dp.K:.2f} C at {m_g:.3f} kg/s, CO2 {T_co2[0] - dp.K:.1f} -> {T_co2[-1] - dp.K:.1f} C "
      f"at {m_c:.3f} kg/s, duty {Q / 1e6:.3f} MW, {N_SEG} segments, p6 {st6['p'] / 1e6:.3f} / p7 {st7['p'] / 1e6:.3f} MPa")
print(f"A_o/A_i = {A_o / A_i:.1f} | finned/plain = {A_o / (math.pi * Do):.1f} | tubes/row {N_t_row:.0f}, tubes in parallel {n_par:.0f}, CO2 G = {G_c:.0f} kg/m2s")
print(f"Fouling R_f'' = {Rf_gas} m2K/W\n")
print("seg  Tc_mean  Tg_mean   Re_g   h_g   eta_o   Re_c    h_c    f_g   f_c   f_w   f_foul  UA_cycle(kW/K)")

rows = []
L_tot = 0.0
ua_cycle = ua_log = 0.0
for i in range(N_SEG):
    Tc, Tg = 0.5 * (T_co2[i] + T_co2[i + 1]), 0.5 * (T_gas[i] + T_gas[i + 1])
    gp = gas_props(Tg)
    Vmax = m_g / (gp["rho"] * A_min)
    Re_g = gp["rho"] * Vmax * Do / gp["mu"]; Pr_g = gp["cp"] * gp["mu"] / gp["k"]
    h_g = 0.134 * Re_g ** 0.681 * Pr_g ** (1 / 3) * (s_fin / hf) ** 0.2 * (s_fin / tf) ** 0.1134 * gp["k"] / Do
    eta_o = fin_eff(h_g)
    hc, pc = 0.5 * (h_c_b[i] + h_c_b[i + 1]), 0.5 * (p_c_b[i] + p_c_b[i + 1])
    rho_c, mu_c = CP.PropsSI("D", "H", hc, "P", pc, "CO2"), CP.PropsSI("V", "H", hc, "P", pc, "CO2")
    k_c, cp_c = CP.PropsSI("L", "H", hc, "P", pc, "CO2"), CP.PropsSI("C", "H", hc, "P", pc, "CO2")
    Re_c = G_c * Di / mu_c; Pr_c = cp_c * mu_c / k_c
    fr = (0.79 * math.log(Re_c) - 1.64) ** -2
    h_c = (fr / 8) * (Re_c - 1000) * Pr_c / (1 + 12.7 * math.sqrt(fr / 8) * (Pr_c ** (2 / 3) - 1)) * k_c / Di
    R_g, R_f, R_c = 1 / (eta_o * h_g * A_o), Rf_gas / (eta_o * A_o), 1 / (h_c * A_i)
    R = R_g + R_f + R_wall + R_c                                   # K.m/W per metre of tube
    # segment conductance exactly as in the cycle model: duty / arithmetic mean temperature difference
    dT1, dT2 = T_gas[i] - T_co2[i], T_gas[i + 1] - T_co2[i + 1]
    Q_i = Q * (f_seg[i + 1] - f_seg[i])
    UA_i = Q_i / (0.5 * (dT1 + dT2))
    lmtd = (dT2 - dT1) / math.log(dT2 / dT1) if abs(dT2 - dT1) > 1e-6 else dT1
    ua_cycle += UA_i; ua_log += Q_i / lmtd
    L_tot += UA_i * R                                              # tube length of the segment, m (all tubes)
    rows.append({"seg": i + 1, "f_from": f_seg[i], "f_to": f_seg[i + 1], "Q_i (W)": Q_i, "T_gas_mean (C)": Tg - dp.K,
                 "T_co2_mean (C)": Tc - dp.K, "UA_i (W/K)": UA_i, "Re_g": Re_g, "h_g (W/m2K)": h_g, "eta_o": eta_o,
                 "Re_c": Re_c, "h_c (W/m2K)": h_c, "f_g": R_g / R, "f_c": R_c / R, "f_w": R_wall / R, "f_foul": R_f / R})
    print(f"{i + 1:3d} {Tc - dp.K:7.1f} {Tg - dp.K:8.1f} {Re_g:7.0f} {h_g:5.1f} {eta_o:6.3f} {Re_c:8.0f} {h_c:6.0f} "
          f"{R_g / R:5.2f} {R_c / R:5.2f} {R_wall / R:5.2f} {R_f / R:6.2f} {UA_i / 1e3:9.2f}")

df = pd.DataFrame(rows)
fg, fc, fw, ff = (df[c].mean() for c in ("f_g", "f_c", "f_w", "f_foul"))
print(f"\nMean shares: f_g = {fg:.3f}, f_c = {fc:.3f}, f_wall = {fw:.3f}, f_foul = {ff:.3f}  (UA-weighted: "
      f"{np.average(df['f_g'], weights=df['UA_i (W/K)']):.3f} / {np.average(df['f_c'], weights=df['UA_i (W/K)']):.3f})")
print(f"UA of the 20 segments, cycle definition (arithmetic mean dT): {ua_cycle / 1e6:.4f} MW/K; with log-mean dT: {ua_log / 1e6:.4f} MW/K; "
      f"design_point value: {dp.heater_design(res, hs)['UA'] / 1e6:.4f} MW/K")
print(f"Tube length {L_tot:.0f} m -> {L_tot / (N_t_row * L_tube):.0f} rows of {N_t_row:.0f} tubes x {L_tube} m; "
      f"internal area {L_tot * A_i:.0f} m2, external {L_tot * A_o:.0f} m2")
print(f"Gas Re range {df['Re_g'].min():.0f} - {df['Re_g'].max():.0f} (Briggs & Young: about 1100-18000)")
name = "heater_resistance_shares" + (f"_{args.tag}" if args.tag else "") + ".csv"
df.to_csv(dp.TABLES / name, index=False)
print(f"Wrote {dp.TABLES / name}")
