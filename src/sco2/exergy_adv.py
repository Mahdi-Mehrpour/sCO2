"""Exergy analysis of the advanced recompression cycle (self-contained).

Input: the dictionary returned by recompression_adv().
All values are per 1 kg of CO2 through the turbine, in J/kg (SI).
Flow exergy: ex = (h - h0) - T0 * (s - s0), with the dead state (T0, p0).

Flows: states 1-3 carry (1-x), state 4 carries x, all others carry 1.

Heater assumption: heat comes from a constant-temperature source at T_source.
If T_source is higher than T7, the heater destruction grows (this is where a
real waste-heat source differs from the ideal case)."""
from CoolProp.CoolProp import PropsSI


def flow_exergy(fluid, state, T0, p0):
    """Physical flow exergy of one state (J/kg)."""
    h0 = PropsSI("H", "T", T0, "P", p0, fluid)
    s0 = PropsSI("S", "T", T0, "P", p0, fluid)
    return (state["h"] - h0) - T0 * (state["s"] - s0)


def exergy_analysis_adv(res, fluid="CO2", T0=298.15, p0=101325.0, T_source=None):
    st = res["states"]
    x = res["x"]
    ex = {k: flow_exergy(fluid, v, T0, p0) for k, v in st.items()}
    if T_source is None:
        T_source = st[7]["T"]          # source at the turbine inlet temperature


    # Exergy of the heat supplied to the heater
    ex_heat = res["q_in"] * (1.0 - T0 / T_source)

    d = {}
    d["Main compressor"] = (1 - x) * (res["w_mc"] - (ex[2] - ex[1]))
    d["Recompressor"] = x * (res["w_rc"] - (ex[4] - ex[10]))
    d["Turbine"] = (ex[7] - ex[8]) - res["w_t"]
    d["HTR"] = (ex[8] - ex[9]) - (ex[6] - ex[5])
    d["LTR"] = (ex[9] - ex[10]) - (1 - x) * (ex[3] - ex[2])
    d["Mixer"] = (1 - x) * ex[3] + x * ex[4] - ex[5]
    d["Heater"] = ex_heat - (ex[7] - ex[6])
    # Cooler: the heat goes to the environment at T0, so the exergy
    # of the stream that is cooled down is lost completely.
    d["Cooler"] = (1 - x) * (ex[10] - ex[1])

    total = sum(d.values())
    return {"destruction": d, "total": total, "ex_heat": ex_heat,
            "eta_ex": res["w_net"] / ex_heat, "T_source": T_source,
            "flow_exergy": ex}
