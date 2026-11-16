"""Cooling system options of the split-flow plant.

ALL VALUES ARE PLACEHOLDERS (typical orders of magnitude, no reference, no
site data). The cooling system sets the compressor inlet temperature T1:
    T1 = coolant inlet temperature + approach, never below T1_floor.
Coolant inlet: dry-bulb temperature (dry air cooler) or wet-bulb temperature
+ tower approach (water loop with a wet cooling tower)."""
import numpy as np
from CoolProp.CoolProp import PropsSI

K = 273.15
CP_AIR, R_AIR = 1005.0, 287.05          # J/kg.K

COOLING = {
    "T1_floor_C": 32.0,                 # C, critical temperature + 1 K
    "dry_air": {
        "label": "Dry air cooler",
        "T_in_C": 25.0,                 # C, design ambient dry-bulb temperature
        "approach": 10.0,               # K, CO2 outlet minus coolant inlet
        "coolant_dT": 10.0,             # K, air temperature rise
        "fan_dp": 200.0,                # Pa
        "eta_fan": 0.65,
    },
    "wet_tower": {
        "label": "Water loop with wet cooling tower",
        "T_wb_C": 18.0,                 # C, design wet-bulb temperature
        "tower_approach": 4.0,          # K, cooling water minus wet bulb
        "approach": 5.0,                # K, CO2 outlet minus cooling-water inlet
        "coolant_dT": 8.0,              # K, water temperature rise
        "fan_pump_fraction": 0.01,      # fan and pump power / cooler duty
        "latent": 2.4e6,                # J/kg, latent heat of water
        "makeup_factor": 1.33,          # makeup water / evaporated water
    },
}


def coolant_inlet_C(option):
    """Temperature of the coolant entering the cooler (C)."""
    c = COOLING[option]
    if option == "dry_air":
        return c["T_in_C"]
    if option == "wet_tower":
        return c["T_wb_C"] + c["tower_approach"]
    raise ValueError("option must be 'dry_air' or 'wet_tower'")


def compressor_inlet_T(option):
    """Compressor inlet temperature (K) set by the cooling system."""
    T1_C = coolant_inlet_C(option) + COOLING[option]["approach"]
    return max(T1_C, COOLING["T1_floor_C"]) + K


def cooler_performance(option, res, hs, n=30, fluid="CO2"):
    """Cooler of the split-flow cycle (state 10 -> state 1), counterflow.
    The coolant enters at the CO2 outlet end and warms up linearly with the
    heat absorbed. UA is the sum of dQ/dT over n segments of equal duty.
    res: splitflow_adv() result; hs: couple_cascade() result."""
    cfg = COOLING[option]
    s1, s10 = res["states"][1], res["states"][10]
    Q = hs["m_co2"] * res["q_out"]                          # W
    f = np.linspace(0.0, 1.0, n + 1)                        # 0 = CO2 outlet
    h = s1["h"] + f * (s10["h"] - s1["h"])
    p = s1["p"] + f * (s10["p"] - s1["p"])
    T_co2 = np.array([PropsSI("T", "H", hi, "P", pi, fluid) for hi, pi in zip(h, p)])
    T_in = coolant_inlet_C(option) + K
    T_cool = T_in + cfg["coolant_dT"] * f
    dT = T_co2 - T_cool
    i = int(np.argmin(dT))
    UA = (float("inf") if dT.min() <= 0.0
          else float(np.sum((Q / n) / (0.5 * (dT[:-1] + dT[1:])))))
    out = {"Q": Q, "T1": s1["T"], "approach_actual": s1["T"] - T_in,
           "min_dT": float(dT[i]), "f_min": float(f[i]), "UA": UA,
           "Q_axis": f * Q, "T_co2": T_co2, "T_cool": T_cool}
    if option == "dry_air":
        rho = 101325.0 / (R_AIR * T_in)
        V = Q / (CP_AIR * cfg["coolant_dT"]) / rho          # m3/s of air
        out.update(P_aux=V * cfg["fan_dp"] / cfg["eta_fan"], flow=V,
                   flow_name="Air flow (m3/s)")
    else:
        out.update(P_aux=cfg["fan_pump_fraction"] * Q,
                   flow=Q / cfg["latent"] * cfg["makeup_factor"],
                   flow_name="Makeup water (kg/s)")
    return out