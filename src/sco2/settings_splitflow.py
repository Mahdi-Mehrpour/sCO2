"""Base-case settings shared by all scripts (one place to change them), plus
the limits and exclusions that are listed in the assumptions table."""

# Economic base case: discount rate (1/yr), economic life (yr), electricity
# value (USD/kWh; placeholder, replace with the site value)
ECON_BASE = {"r": 0.08, "N": 20, "c_e": 0.08}

HEATER_POSITION = ("Heater in the primary off-gas line after post-combustion "
                   "(hot-duct position), upstream of water-cooled duct and "
                   "evaporative cooling; stable gas condition")

# Limits come from the list supplied for this project and are NOT verified.
# key: which model value is compared with the limit (None = listed only)
LIMITS = [
    dict(name="Maximum cycle pressure (250 bar, at the compressor outlet)",
         value=25.0, unit="MPa", kind="hard", key="p_max",
         basis="STEP turbine operates at 250 bar",
         enforced="enforced: pressure ratio is capped"),
    dict(name="Design pressure relative to operating pressure",
         value=1.1, unit="-", kind="info", key=None,
         basis="STEP: 280 bar design for 250 bar operation; a precedent, not a code rule",
         enforced="not used in the calculation"),
    dict(name="Turbine inlet temperature",
         value=700.0, unit="C", kind="validity", key="T7",
         basis="upper end of the radial turbine cost data (Weiland et al.)",
         enforced="turbine inlet is fixed at 550 C"),
    dict(name="HTR hot inlet temperature (state 8), base case",
         value=550.0, unit="C", kind="hard", key="T8",
         basis="316 stainless steel, conservative value",
         enforced="reported; not yet a constraint in the optimisation"),
    dict(name="HTR hot inlet temperature (state 8), sensitivity case",
         value=600.0, unit="C", kind="sensitivity", key="T8",
         basis="STEP HTR hot inlet in 316 stainless steel",
         enforced="reported only"),
    dict(name="Recuperator precedent",
         value="575 C / 280 bar", unit="", kind="info", key=None,
         basis="Thar Energy recuperator, ASME Sec. VIII Div. 1",
         enforced="not used in the calculation"),
    dict(name="Heater material above 700 C",
         value="Ni alloy (740H or 617)", unit="", kind="info", key=None,
         basis="STEP heater; the CO2 side stays at 550 C in this model",
         enforced="not used in the calculation"),
]

NOT_MODELLED = [
    "Gas ducts, dust handling and filters, controls, grid connection, engineering and contingency (costs)",
    "Heat buffer (thermal storage) between the gas and the cycle (cost and dynamics)",
    "Pressure effect on equipment cost (the correlations have no pressure factor)",
    "Taxes, financing structure and incentives",
    "Changes of gas composition; dust and fouling on the gas side",
    "Dynamic behaviour and off-design operation of turbomachinery and heat exchangers",
    "Mechanical design of the heaters (wall temperature, tube material)",
]