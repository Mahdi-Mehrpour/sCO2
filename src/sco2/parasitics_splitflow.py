"""Parasitic loads and net electrical output of the split-flow plant.

ALL VALUES ARE PLACEHOLDERS (typical orders of magnitude, no reference).
Powers in W. Cycle results are per kg of turbine flow; absolute values use
the CO2 flow from the heat-source coupling.

drive = "motor": turbine -> gearbox -> generator, and an electric motor ->
                 gearbox -> compressor.
drive = "shaft": compressor on the turbine shaft; one gearbox and one
                 generator on the net shaft power."""

PARASITIC = {
    "drive": "motor",
    "eta_gearbox": 0.985,
    "eta_generator": 0.975,
    "eta_motor": 0.955,
    "T_ambient_C": 25.0,        # C, design ambient air temperature
    "air_dT": 10.0,             # K, air temperature rise in the cooler
    "fan_dp_air": 200.0,        # Pa, air-side fan pressure rise
    "eta_fan_air": 0.65,
    "gas_dp_total": 3000.0,     # Pa, gas-side pressure drop (heaters + ducts)
    "eta_fan_gas": 0.70,
}
CP_AIR = 1005.0                # J/kg.K
R_AIR = 287.05                 # J/kg.K
R_UNIV = 8314.46               # J/kmol.K


def parasitic_loads(res, hs, gas, m_gas, p_gas, par=None):
    """res: splitflow_adv() result; hs: couple_cascade() result;
    gas: GasStream; m_gas: gas flow (kg/s); p_gas: gas pressure (Pa)."""
    p = PARASITIC if par is None else par
    m = hs["m_co2"]
    W_t, W_c = m * res["w_t"], m * res["w_mc"]
    Q_cool = m * res["q_out"]
    gb, gen, mot = p["eta_gearbox"], p["eta_generator"], p["eta_motor"]

    if p["drive"] == "motor":
        P_gen = W_t * gb * gen
        P_motor = W_c / (gb * mot)
        loss_gb = W_t * (1 - gb) + P_motor * mot * (1 - gb)
        loss_gen = W_t * gb * (1 - gen)
        loss_mot = P_motor * (1 - mot)
    elif p["drive"] == "shaft":
        W_ns = W_t - W_c
        P_gen = W_ns * gb * gen
        P_motor = 0.0
        loss_gb = W_ns * (1 - gb)
        loss_gen = W_ns * gb * (1 - gen)
        loss_mot = 0.0
    else:
        raise ValueError("drive must be 'motor' or 'shaft'")

    # Cooler fans: air flow from the duty and the air temperature rise
    rho_air = 101325.0 / (R_AIR * (p["T_ambient_C"] + 273.15))
    V_air = Q_cool / (CP_AIR * p["air_dT"]) / rho_air          # m3/s
    P_fan_air = V_air * p["fan_dp_air"] / p["eta_fan_air"]

    # Off-gas fan at the cold end (stack temperature)
    rho_gas = p_gas * gas.M / (R_UNIV * hs["T_out"])
    V_gas = m_gas / rho_gas
    P_fan_gas = V_gas * p["gas_dp_total"] / p["eta_fan_gas"]

    net_el = P_gen - P_motor - P_fan_air - P_fan_gas
    net_shaft = W_t - W_c                       # what the cycle model reports
    closure = (net_shaft - loss_gb - loss_gen - loss_mot
               - P_fan_air - P_fan_gas - net_el)
    return {"W_turbine": W_t, "W_compressor": W_c, "net_shaft": net_shaft,
            "loss_gearbox": loss_gb, "loss_generator": loss_gen,
            "loss_motor": loss_mot, "fan_air": P_fan_air, "fan_gas": P_fan_gas,
            "V_air": V_air, "V_gas": V_gas, "net_el": net_el,
            "closure": closure,
            "eta_net_el": net_el / (m * (res["q_in"] + res["q_b"]))}