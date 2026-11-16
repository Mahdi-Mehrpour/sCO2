"""Annual operating pattern of the plant (EAF heat cycle).

ALL VALUES ARE PLACEHOLDERS (no plant data). Each phase of one furnace heat
has a share of the heat time, an off-gas temperature and an off-gas flow
relative to the design flow (EAF["m_gas"]).

With a stable gas condition (current base case) only days_per_year,
hours_per_day and plant_availability are used; the phases are kept for the
later treatment of gas temperature fluctuations."""

OPERATION = {
    "days_per_year": 330.0,         # furnace operating days (outages excluded)
    "hours_per_day": 24.0,
    "plant_availability": 0.90,     # availability of the sCO2 unit itself
    "max_load": 1.0,                # CO2 flow cap relative to the design flow
    "phases": [
        {"name": "Power-off, tapping, charging", "time_fraction": 0.12,
         "T_gas_C": 350.0, "flow_factor": 0.4},
        {"name": "Early melting", "time_fraction": 0.20,
         "T_gas_C": 650.0, "flow_factor": 0.8},
        {"name": "Main melting", "time_fraction": 0.40,
         "T_gas_C": 850.0, "flow_factor": 1.0},
        {"name": "Refining", "time_fraction": 0.28,
         "T_gas_C": 950.0, "flow_factor": 1.2},
    ],
}


def calendar_hours(op=OPERATION):
    """Hours per year in which the furnace is scheduled to operate."""
    return op["days_per_year"] * op["hours_per_day"]


def check_pattern(op=OPERATION):
    """The time fractions of the phases must add up to 1."""
    total = sum(p["time_fraction"] for p in op["phases"])
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Phase time fractions add up to {total}, not 1")