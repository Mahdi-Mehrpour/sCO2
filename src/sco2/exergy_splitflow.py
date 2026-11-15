"""Exergy analysis of the split-flow cycle coupled to the EAF gas.

Inputs: results of splitflow_adv() and couple_cascade(). SI units.
Cycle quantities are per kg of turbine flow; absolute values (W) use the
CO2 flow m = hs["m_co2"]. Only physical exergy is considered.

System balance: gas exergy in = net power + destruction in all components
+ exergy loss in the cooler + exergy leaving with the stack gas."""
from .exergy_adv import flow_exergy


def exergy_splitflow(res, hs, gas, m_gas, fluid="CO2", T0=298.15, p0=101325.0):
    st, y, m = res["states"], res["y"], hs["m_co2"]
    ex = {k: flow_exergy(fluid, v, T0, p0) for k, v in st.items()}
    ex_mid = m_gas * float(gas.ex(hs["T_gas_mid"]))     # gas between the heaters

    # Exergy destruction in W (cooler = exergy of the cooled stream, lost)
    d = {
        "Main heater": hs["ex_in"] - ex_mid - m * (ex[7] - ex[6]),
        "Branch heater": ex_mid - hs["ex_out"] - m * y * (ex[11] - ex[2]),
        "HTR": m * ((ex[8] - ex[9]) - (ex[6] - ex[5])),
        "LTR": m * ((ex[9] - ex[10]) - (1 - y) * (ex[3] - ex[2])),
        "Mixer": m * ((1 - y) * ex[3] + y * ex[11] - ex[5]),
        "Turbine": m * ((ex[7] - ex[8]) - res["w_t"]),
        "Main compressor": m * (res["w_mc"] - (ex[2] - ex[1])),
        "Cooler": m * (ex[10] - ex[1]),
    }
    # Exergy efficiency = exergy of the product / exergy of the fuel
    eff = {
        "Main heater": m * (ex[7] - ex[6]) / (hs["ex_in"] - ex_mid),
        "Branch heater": m * y * (ex[11] - ex[2]) / (ex_mid - hs["ex_out"]),
        "HTR": (ex[6] - ex[5]) / (ex[8] - ex[9]),
        "LTR": (1 - y) * (ex[3] - ex[2]) / (ex[9] - ex[10]),
        "Mixer": ex[5] / ((1 - y) * ex[3] + y * ex[11]),
        "Turbine": res["w_t"] / (ex[7] - ex[8]),
        "Main compressor": (ex[2] - ex[1]) / res["w_mc"],
        "Cooler": None,
    }
    W = m * res["w_net"]
    stack = hs["ex_out"]
    balance = hs["ex_in"] - W - sum(d.values()) - stack
    ex_to_co2 = m * ((ex[7] - ex[6]) + y * (ex[11] - ex[2]))
    return {"flow_exergy": ex, "destruction_W": d, "exergy_eff": eff,
            "W": W, "stack_W": stack, "balance_W": balance,
            "ex_to_co2_W": ex_to_co2, "ex_mid_W": ex_mid}