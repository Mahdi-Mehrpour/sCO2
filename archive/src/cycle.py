"""Simple recuperated sCO2 Brayton cycle.

States: 1 compressor inlet | 2 compressor outlet | 3 recuperator cold outlet
        4 turbine inlet    | 5 turbine outlet    | 6 recuperator hot outlet"""
from .components import state_Tp, compressor, turbine, recuperator


def recuperated_brayton(fluid="CO2", T1=305.15, p1=7.7e6, pr=2.6,
                        T4=823.15, eta_c=0.85, eta_t=0.90, eps=0.90):
    # State 1: compressor inlet (fixed by T1, p1)
    s1 = state_Tp(fluid, T1, p1)
    # State 2: compressor outlet at pressure p1 * pr
    s2, w_c = compressor(fluid, s1, p1 * pr, eta_c)
    # State 4: turbine inlet (fixed by T4 and the high pressure)
    s4 = state_Tp(fluid, T4, p1 * pr)
    # State 5: turbine outlet, expanded back to the low pressure p1
    s5, w_t = turbine(fluid, s4, p1, eta_t)
    # States 3 and 6: recuperator (hot side = turbine exhaust, cold side = compressor outlet)
    s3, s6, q_rec = recuperator(fluid, s5, s2, eps)

    q_in = s4["h"] - s3["h"]      # Heater (heat input from the heat source)
    q_out = s6["h"] - s1["h"]     # Cooler (heat rejected to the environment)
    w_net = w_t - w_c             # Net specific work
    return {"states": {1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s6},
            "w_c": w_c, "w_t": w_t, "w_net": w_net,
            "q_in": q_in, "q_out": q_out, "q_rec": q_rec,
            "eta": w_net / q_in}   # Thermal efficiency = net work / heat input