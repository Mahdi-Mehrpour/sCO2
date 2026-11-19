"""Exergy analysis of the split-flow cycle coupled to the EAF gas.

Inputs: results of splitflow_adv() and couple_cascade(). SI units.
Cycle quantities are per kg of turbine flow; absolute values (W) use the
CO2 flow m = hs["m_co2"]. The balances use the PHYSICAL exergy of the gas and
the CO2; the chemical exergy of the CO in the off-gas is reported separately
(it is not recoverable in this plant, the CO is not burned).

exergy_splitflow(): cycle components, heaters and stack, on a net shaft
power basis.
exergy_plant(): adds the parasitic losses (gearboxes, generator, motor,
fans), so the balance closes on net electrical power:
  gas exergy in = net electrical power + all destruction + cooler loss
                  + parasitic losses + exergy leaving with the stack gas.

Three exergy efficiencies of the plant are returned:
  eta_system        = net electrical / physical exergy of the gas in
                      (proportional to net power for a given gas)
  eta_fuel_product  = net electrical / (gas exergy in - stack exergy), i.e. the
                      exergy actually taken from the gas
  eta_total         = net electrical / (physical + chemical exergy of the CO)"""
import numpy as np
from CoolProp.CoolProp import PropsSI

R_UNIVERSAL = 8.314462618      # J/(mol K)
EX_CH_CO = 275.1e3             # J/mol, standard chemical exergy of CO (Szargut)


def flow_exergy(fluid, state, T0, p0):
    """Physical flow exergy of one state (J/kg): (h - h0) - T0 (s - s0)."""
    h0 = PropsSI("H", "T", T0, "P", p0, fluid)
    s0 = PropsSI("S", "T", T0, "P", p0, fluid)
    return (state["h"] - h0) - T0 * (state["s"] - s0)


def chemical_exergy_CO(gas, X, m_gas, T0=298.15):
    """Chemical exergy flow (W) of the CO in the off-gas (oxidation to CO2),
    including its dilution term: x_CO (eps_CO + R T0 ln x_CO) per mol of gas.
    X: mole fractions of the gas (dictionary); gas.M in kg/kmol (= g/mol)."""
    x = X.get("CO", 0.0)
    if x <= 0.0:
        return 0.0
    per_mol = x * (EX_CH_CO + R_UNIVERSAL * T0 * np.log(x))     # J per mol of gas
    return m_gas * per_mol / (gas.M * 1e-3)


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


def exergy_plant(P, gas, m_gas, fluid="CO2", T0=298.15, p0=101325.0, X=None):
    """Exergy analysis of the whole plant with parasitic loads.
    P: dictionary from evaluate_plant() (cycle, gas coupling, parasitic loads,
    cooler and net electrical power).
    Fan work is counted as fully lost; machines by their efficiency.
    X: gas mole fractions (default: the EAF case) for the chemical exergy of CO."""
    if X is None:
        from .heat_source_adv import EAF
        X = EAF["X"]
    res, hs, par, cp = P["res"], P["hs"], P["par"], P["cooler"]
    base = exergy_splitflow(res, hs, gas, m_gas, fluid, T0, p0)
    d = dict(base["destruction_W"])
    d["Gearboxes"] = par["loss_gearbox"]
    d["Generator"] = par["loss_generator"]
    d["Compressor motor"] = par["loss_motor"]
    d["Cooling tower fan and pump"] = cp["P_aux"]
    d["Off-gas fan"] = par["fan_gas"]

    net_el, stack, ex_in = P["net_el"], base["stack_W"], hs["ex_in"]
    balance = ex_in - net_el - sum(d.values()) - stack
    ex_chem = chemical_exergy_CO(gas, X, m_gas, T0)
    return {"flow_exergy": base["flow_exergy"], "destruction_W": d,
            "base_eff": base["exergy_eff"], "stack_W": stack, "net_el": net_el,
            "ex_in": ex_in, "ex_to_co2_W": base["ex_to_co2_W"],
            "balance_W": balance, "eta_system": net_el / ex_in,
            "eta_cycle": net_el / base["ex_to_co2_W"],
            "ex_chem_CO_W": ex_chem,
            "eta_fuel_product": net_el / (ex_in - stack),
            "eta_total": net_el / (ex_in + ex_chem)}
