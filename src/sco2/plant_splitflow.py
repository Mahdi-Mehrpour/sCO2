"""Plant-level evaluation of the split-flow cycle with the EAF gas, wet
cooling tower and parasitic loads. One function for the optimiser and the
tables: evaluate_plant(y, T_b, pr).

Gas condition is stable (EAF["T_in"], EAF["m_gas"]) unless T_in or m_gas are
passed. Operating hours come from the calendar and the unit availability."""
from .cycle_adv import splitflow_adv
from .heat_source_adv import EAF, GasStream, couple_cascade
from .parasitics_splitflow import PARASITIC, parasitic_loads
from .cooling_splitflow import compressor_inlet_T, cooler_performance
from .operation_splitflow import OPERATION, calendar_hours

COOLING_OPTION = "wet_tower"
DRIVE = "motor"
_GAS = []


def get_gas():
    """Gas table, built once."""
    if not _GAS:
        _GAS.append(GasStream(EAF["X"], EAF["p"], T0=298.15))
    return _GAS[0]


def operating_hours(op=OPERATION):
    """Equivalent full-load hours for the stable gas condition (h/yr)."""
    return calendar_hours(op) * op["plant_availability"]


def evaluate_plant(y, T_b, pr, check_pinch=True, option=COOLING_OPTION,
                   drive=DRIVE, T_in=None, m_gas=None, **cycle_kw):
    """Full plant evaluation. T_in (K) and m_gas (kg/s) override the EAF case.
    Returns a dictionary with the cycle result (res), gas coupling (hs),
    parasitic loads (par), cooler and the net electrical output."""
    gas = get_gas()
    dTp, Tmin = EAF["dT_pinch_heater"], EAF["T_stack_min"]
    T_in = EAF["T_in"] if T_in is None else T_in
    mg = EAF["m_gas"] if m_gas is None else m_gas
    T1 = compressor_inlet_T(option)
    res = splitflow_adv(T1=T1, y=y, T_b=T_b, pr=pr, check_pinch=check_pinch,
                        **cycle_kw)
    hs = couple_cascade(res, gas, T_in, mg, dTp, dTp, Tmin)
    par = parasitic_loads(res, hs, gas, mg, EAF["p"], dict(PARASITIC, drive=drive))
    cooler = cooler_performance(option, res, hs)

    # parasitic_loads() assumes dry air fans; swap in the cooler of this option
    net_el = par["net_el"] + par["fan_air"] - cooler["P_aux"]
    q_heat = hs["m_co2"] * (res["q_in"] + res["q_b"])
    hours = operating_hours()
    return {"res": res, "hs": hs, "par": par, "cooler": cooler,
            "net_el": net_el, "q_heat": q_heat, "eta_el": net_el / q_heat,
            "hours": hours, "energy_GWh": net_el / 1e6 * hours / 1e3,
            "T1": T1, "T_in": T_in, "m_gas": mg}