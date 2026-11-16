"""Capital cost of the split-flow cycle: sizes the components from the cycle
results and adds up their costs (equations are in cost_components.py)."""
from .cost_components import (COST, ECON, pressure_factor, ua_from_profile,
                              cost_turbine, cost_compressor, cost_generator,
                              cost_recuperator, cost_heater, cost_cooler,
                              annual_economics)


def capital_cost(res, hs, cost=None, n_p=0.6):
    """res: dictionary from splitflow_adv() (with pinch data);
    hs: dictionary from couple_cascade().
    Returns the cost of each component, the total, net power and specific cost."""
    c = COST if cost is None else cost
    m = hs["m_co2"]                                  # kg/s of CO2 (turbine flow)
    f_p = pressure_factor(res["states"][2]["p"], n_p)

    # Component sizes
    W_t = m * res["w_t"] / 1e3                       # kW, turbine
    W_c = m * res["w_mc"] / 1e3                      # kW, compressor
    W_net = m * res["w_net"] / 1e3                   # kW, net
    Q_cool = m * res["q_out"] / 1e3                  # kW, cooler duty
    UA = {
        "HTR": m * res["pinch"]["HTR"]["UA"],
        "LTR": m * res["pinch"]["LTR"]["UA"],
        "Main heater": ua_from_profile(hs["Q1"], hs["T_gas1"], hs["T_co2_1"]),
        "Branch heater": ua_from_profile(hs["Q2"], hs["T_gas2"], hs["T_co2_2"]),
    }

    # Component costs
    items = {
        "Turbine": cost_turbine(W_t, f_p, c),
        "Main compressor": cost_compressor(W_c, f_p, c),
        "Generator": cost_generator(W_t, c),
        "HTR": cost_recuperator(UA["HTR"], f_p, c),
        "LTR": cost_recuperator(UA["LTR"], f_p, c),
        "Main heater": cost_heater(UA["Main heater"], f_p, c),
        "Branch heater": cost_heater(UA["Branch heater"], f_p, c),
        "Cooler": cost_cooler(Q_cool, c),
    }
    total = float(sum(items.values()))
    return {"items": items, "UA": UA, "total": total, "W_net_kW": W_net,
            "specific": total / W_net, "pressure_factor": f_p}


def economics(c, econ=None):
    """Annual economics of a capital_cost() result."""
    return annual_economics(c["total"], c["W_net_kW"],
                            ECON if econ is None else econ)