"""Plant-level evaluation of the split-flow cycle with the EAF gas, wet
cooling tower and parasitic loads. One function for the optimiser and the
tables: evaluate_plant(y, T_b, pr).

Gas condition is stable (EAF["T_in"], EAF["m_gas"]). Operating hours come
from the calendar and the unit availability (no phase pattern)."""
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
                   drive=DRIVE):
    """Full plant evaluation. Returns a dictionary with the cycle result (res),
    gas coupling (hs), parasitic loads (par), cooler (cooler) and the net
    electrical output."""
    gas = get_gas()
    dTp, Tmin, mg = EAF["dT_pinch_heater"], EAF["T_stack_min"], EAF["m_gas"]
    T1 = compressor_inlet_T(option)
    res = splitflow_adv(T1=T1, y=y, T_b=T_b, pr=pr, check_pinch=check_pinch)
    hs = couple_cascade(res, gas, EAF["T_in"], mg, dTp, dTp, Tmin)
    par = parasitic_loads(res, hs, gas, mg, EAF["p"], dict(PARASITIC, drive=drive))
    cooler = cooler_performance(option, res, hs)

    # parasitic_loads() assumes dry air fans; swap in the cooler of this option
    net_el = par["net_el"] + par["fan_air"] - cooler["P_aux"]
    q_heat = hs["m_co2"] * (res["q_in"] + res["q_b"])
    hours = operating_hours()
    return {"res": res, "hs": hs, "par": par, "cooler": cooler,
            "net_el": net_el, "q_heat": q_heat, "eta_el": net_el / q_heat,
            "hours": hours, "energy_GWh": net_el / 1e6 * hours / 1e3,
            "T1": T1}