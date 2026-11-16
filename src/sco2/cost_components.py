"""Cost equations of the individual components (equipment cost only).

One function per component, like components.py for the thermodynamics.
ALL COEFFICIENTS ARE PLACEHOLDERS (order of magnitude, no reference).
Replace them with literature values before reporting absolute numbers.
Costs are in USD, powers in kW, UA in W/K.

Pressure enters through one factor on the high-pressure equipment:
    f_p = (p_high / P_REF) ** n_p
n_p = 0 means pressure has no effect on cost."""
import numpy as np

# ----------------------------------------------------------------------
# Unit costs (placeholders)
# ----------------------------------------------------------------------
COST = {
    "turbine_per_kW": 250.0,        # $/kW of turbine shaft power
    "compressor_per_kW": 200.0,     # $/kW of compressor shaft power
    "generator_per_kW": 100.0,      # $/kW of electrical power
    "recuperator_per_WK": 2.0,      # $/(W/K), CO2-CO2 recuperators
    "heater_per_WK": 2.5,           # $/(W/K), gas-CO2 heaters
    "cooler_per_kWth": 60.0,        # $/kW of heat rejected
}
P_REF = 20e6                        # Pa, reference pressure of f_p

# Economic placeholders (replace with the data of your case)
ECON = {
    "price_per_MWh": 80.0,          # $/MWh, value of the electricity
    "hours_per_year": 6000.0,       # h/yr, operating hours
    "interest": 0.085,              # 1/yr, interest rate
    "lifetime_years": 15.0,         # yr, economic lifetime
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def pressure_factor(p_high, n_p, p_ref=P_REF):
    """Cost multiplier of high-pressure equipment: (p / p_ref) ** n_p."""
    return (p_high / p_ref) ** n_p


def ua_from_profile(Q, T_hot, T_cold):
    """UA (W/K) of a heat exchanger from its temperature profile.
    Q: total duty (W). T_hot, T_cold: arrays of n+1 temperatures (K) at the
    boundaries of n segments of equal duty. UA = sum of (dQ / dT_segment)."""
    dT = np.asarray(T_hot) - np.asarray(T_cold)
    if dT.min() <= 0.0:
        return float("inf")
    dT_mid = 0.5 * (dT[:-1] + dT[1:])
    return float(np.sum((Q / (len(dT) - 1)) / dT_mid))


def capital_recovery_factor(interest, lifetime_years):
    """CRF = i (1+i)^N / ((1+i)^N - 1), in 1/yr."""
    f = (1.0 + interest) ** lifetime_years
    return interest * f / (f - 1.0)


# ----------------------------------------------------------------------
# Component cost equations (USD)
# ----------------------------------------------------------------------
def cost_turbine(W_kW, f_p, c=COST):
    """C = c_turbine * W_turbine * f_p"""
    return c["turbine_per_kW"] * W_kW * f_p


def cost_compressor(W_kW, f_p, c=COST):
    """C = c_compressor * W_compressor * f_p"""
    return c["compressor_per_kW"] * W_kW * f_p


def cost_generator(W_kW, c=COST):
    """C = c_generator * W_turbine   (no pressure factor)"""
    return c["generator_per_kW"] * W_kW


def cost_recuperator(UA, f_p, c=COST):
    """C = c_recuperator * UA * f_p"""
    return c["recuperator_per_WK"] * UA * f_p


def cost_heater(UA, f_p, c=COST):
    """C = c_heater * UA * f_p   (gas-to-CO2 heat exchanger)"""
    return c["heater_per_WK"] * UA * f_p


def cost_cooler(Q_kW, c=COST):
    """C = c_cooler * Q_cooler   (low pressure side, no pressure factor)"""
    return c["cooler_per_kWth"] * Q_kW


# ----------------------------------------------------------------------
# Annual economics
# ----------------------------------------------------------------------
def annual_economics(C_total, W_net_kW, econ=ECON):
    """Revenue = price * net power * hours
    Annualised capital = CRF * C_total
    Net benefit = revenue - annualised capital
    Returns amounts in M$/yr, simple payback (yr, no O&M) and the capital
    charge per MWh (USD/MWh)."""
    crf = capital_recovery_factor(econ["interest"], econ["lifetime_years"])
    W_MW = W_net_kW / 1e3
    energy = W_MW * econ["hours_per_year"]                     # MWh/yr
    revenue = econ["price_per_MWh"] * energy / 1e6             # M$/yr
    annual_cap = crf * C_total / 1e6                           # M$/yr
    return {"crf": crf, "revenue": revenue, "annual_capital": annual_cap,
            "benefit": revenue - annual_cap,
            "payback": C_total / 1e6 / revenue if revenue > 0 else float("inf"),
            "lcoe_capital": crf * C_total / energy if energy > 0 else float("inf")}