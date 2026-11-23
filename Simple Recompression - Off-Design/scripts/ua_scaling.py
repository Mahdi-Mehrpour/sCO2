"""Flow dependence of the UA of a fixed heat exchanger (an assumption, kept in one place).

Each side's film resistance scales with the flow of its own side, h ~ m**n with n about 0.8 for turbulent flow
(Dittus-Boelter); the wall resistance is neglected. With rshare_hot the fraction of the design resistance that sits on
the hot side (1 - rshare_hot on the cold side) and r = flow / design flow:

    UA / UA0 = 1 / ( rshare_hot * r_hot**-n + (1 - rshare_hot) * r_cold**-n )

rshare_hot = 0.5: equal resistances (recuperators). A gas heater is dominated by the gas side (low gas-side
coefficient), so rshare_hot is close to 1 there: its UA then follows the gas flow. The shares and the exponent
are placeholders until vendor data are available."""


def ua_ratio(r_hot, r_cold, rshare_hot=0.5, n=0.8):
    """UA / UA0 for flow ratios r_hot, r_cold (flow / design flow); equal to 1 at r = 1 for any share."""
    return 1.0 / (rshare_hot * r_hot ** -n + (1.0 - rshare_hot) * r_cold ** -n)
