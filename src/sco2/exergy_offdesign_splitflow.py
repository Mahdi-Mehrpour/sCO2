"""Exergy account of ONE off-design operating point (frozen hardware).

Same conventions as exergy_splitflow.py (physical exergy, dead state T0, p0):
  gas exergy in = net electrical power + destruction in the heat exchangers,
                  recuperators, mixer, turbine, compressor + cooler loss
                  + electromechanical and auxiliary losses
                  + exergy of the bypassed gas + exergy of the gas leaving
                  the heaters.
pt: dictionary returned by offdesign_point() (or the 'point' of operate())."""
from .exergy_splitflow import flow_exergy


def exergy_point(pt, gas, T_gas, m_gas, fluid="CO2", T0=298.15, p0=101325.0):
    """All values in W. Returns the components, the loss groups used in the
    report, the fuel (exergy taken from the gas) and the balance error."""
    st, y, m, m_b = pt["states"], pt["y"], pt["m_co2"], pt["m_bypass"]
    m_g = m_gas - m_b                                  # gas through the heaters
    ex = {k: flow_exergy(fluid, v, T0, p0) for k, v in st.items()}

    def gx(T):
        return float(gas.ex(T))

    T_mid = float(gas.T_of_h(float(gas.h(T_gas)) - pt["Q_mh"] / max(m_g, 1e-9)))
    T_out = pt["T_gas_out_heaters"]
    ex_in = m_gas * gx(T_gas)
    D = {
        "Main heater": m_g * (gx(T_gas) - gx(T_mid)) - m * (ex[7] - ex[6]),
        "Branch heater": m_g * (gx(T_mid) - gx(T_out)) - m * y * (ex[11] - ex[2]),
        "HTR": m * ((ex[8] - ex[9]) - (ex[6] - ex[5])),
        "LTR": m * ((ex[9] - ex[10]) - (1 - y) * (ex[3] - ex[2])),
        "Mixer": m * ((1 - y) * ex[3] + y * ex[11] - ex[5]),
        "Turbine": m * (ex[7] - ex[8]) - pt["W_t"],
        "Main compressor": pt["W_mc"] - m * (ex[2] - ex[1]),
        "Cooler": m * (ex[10] - ex[1]),
    }
    bypass = m_b * gx(T_gas)
    stack = m_g * gx(T_out)
    aux = pt["net_shaft"] - pt["net_el"]               # gearboxes, generator, motor, fans
    W = pt["net_el"]
    balance = ex_in - W - sum(D.values()) - bypass - stack - aux
    return {
        "ex_in": ex_in, "W": W, "components": D, "bypass": bypass, "stack": stack,
        "aux": aux, "balance": balance,
        "fuel": m_g * (gx(T_gas) - gx(T_out)),         # exergy actually taken from the gas
        "groups": {
            "Net electrical power": W,
            "Bypassed gas": bypass,
            "Stack gas": stack,
            "Idle and restart": 0.0,
            "Heat exchangers": (D["Main heater"] + D["Branch heater"] + D["HTR"]
                                + D["LTR"] + D["Mixer"]),
            "Turbine and compressor": D["Turbine"] + D["Main compressor"],
            "Cooler": D["Cooler"],
            "Electromechanical and auxiliary": aux,
        },
    }


GROUPS = ["Net electrical power", "Bypassed gas", "Stack gas", "Idle and restart",
          "Heat exchangers", "Turbine and compressor", "Cooler",
          "Electromechanical and auxiliary"]
