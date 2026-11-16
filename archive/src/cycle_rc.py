"""Recompression sCO2 Brayton cycle (two recuperators, LTR and HTR).

All flows are per 1 kg of total CO2 through the turbine.
x = fraction of the flow sent to the recompressor (bypasses the cooler).

States: 1  main compressor (MC) inlet    | 2  MC outlet
        3  LTR cold outlet               | 4  recompressor (RC) outlet
        5  mixer outlet (HTR cold inlet) | 6  HTR cold outlet
        7  turbine inlet (heater outlet) | 8  turbine outlet
        9  HTR hot outlet (LTR hot inlet)| 10 LTR hot outlet (flow split point)"""
from .components import (state_Tp, compressor, turbine, recuperator,
                         recuperator_flows, mixer)


def recompression_brayton(fluid="CO2", T1=305.15, p1=7.7e6, pr=2.6,
                          T7=823.15, eta_mc=0.85, eta_rc=0.85, eta_t=0.90,
                          eps_ltr=0.90, eps_htr=0.90, x=0.3):
    p_high = p1 * pr

    # Main compressor (carries 1-x of the flow)
    s1 = state_Tp(fluid, T1, p1)
    s2, w_mc = compressor(fluid, s1, p_high, eta_mc)

    # Turbine (carries the full flow)
    s7 = state_Tp(fluid, T7, p_high)
    s8, w_t = turbine(fluid, s7, p1, eta_t)

    # The recuperators and the recompressor depend on each other through the
    # split point (state 10), but the hot side can be computed first:
    # HTR needs state 5 (cold inlet), which needs states 3 and 4, which need
    # state 10. So we iterate until state 5 stops changing.
    s5 = s2   # initial guess for the HTR cold inlet
    for _ in range(100):
        s6, s9, q_htr = recuperator(fluid, s8, s5, eps_htr)
        s3, s10, q_ltr = recuperator_flows(fluid, s9, s2, eps_ltr, 1 - x)
        s4, w_rc = compressor(fluid, s10, p_high, eta_rc)
        s5_new = mixer(fluid, s3, 1 - x, s4, x)
        if abs(s5_new["h"] - s5["h"]) < 1e-3:   # J/kg
            s5 = s5_new
            break
        s5 = s5_new

    q_in = s7["h"] - s6["h"]                 # heater
    q_out = (1 - x) * (s10["h"] - s1["h"])   # cooler (only the 1-x stream)
    w_net = w_t - (1 - x) * w_mc - x * w_rc
    return {"states": {1: s1, 2: s2, 3: s3, 4: s4, 5: s5,
                       6: s6, 7: s7, 8: s8, 9: s9, 10: s10},
            "w_mc": w_mc, "w_rc": w_rc, "w_t": w_t, "w_net": w_net,
            "q_in": q_in, "q_out": q_out, "q_htr": q_htr, "q_ltr": q_ltr,
            "x": x, "eta": w_net / q_in}
