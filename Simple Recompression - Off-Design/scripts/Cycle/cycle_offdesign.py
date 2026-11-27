"""Integrated off-design model of the simple recompression cycle with the component models of the off-design study.

Layout and states (as in the design study): 1 main compressor (MC) inlet = cooler outlet; 2 MC outlet; 3 LTR cold outlet;
4 recompressor (RC) outlet; 5 mixer outlet = HTR cold inlet; 6 HTR cold outlet = heater inlet; 7 heater outlet = turbine inlet;
8 turbine outlet = HTR hot inlet; 9 HTR hot outlet = LTR hot inlet; 10 LTR hot outlet, where the flow splits into the RC branch
(x m) and the cooler branch ((1 - x) m, to state 1).

Components: radial turbine at constant speed (swallowing capacity), MC and RC from the Sandia map at given speed ratios,
segmented gas heater (gas inlet temperature and flow, bypass damper), segmented HTR and LTR, water-cooled PCHE with a wet cooling
tower (wet-bulb temperature, flow factor of the water and tower air at constant L/G).

Inputs (operating point): gas inlet temperature and flow, wet-bulb temperature, MC and RC speed ratios N / N_des, cooler flow factor,
gas bypass fraction, MC inlet pressure p1 (the inventory control variable).
Unknowns (7): CO2 flow m, recompression fraction x, turbine outlet pressure p8, LTR hot inlet pressure p9, and the enthalpies h8, h9,
h10. Equations: turbine swallowing capacity = m; RC outlet pressure = LTR cold outlet pressure (mixing); the loop closures of
h8 (turbine outlet), h9 (HTR hot outlet), p9 (HTR hot outlet), h10 (LTR hot outlet); and the low-pressure chain
p10 (LTR hot outlet) = p10 (cooler inlet), which fixes p8 through p1. Solved with scipy root (hybr) on scaled unknowns, starting from
the design point (or the previous solution)."""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import root

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
for sub in ("Cooler", "LTR & HTR", "Gas Heater", "Compressor", "Turbine"):
    sys.path.insert(0, str(SCRIPTS / sub))
sys.path.insert(0, str(SCRIPTS))

import pandas as pd

import compressor as cmp_
import heater as ht
import recuperator as rcp
import turbine as trb
import water_cooler as wc
from cooling_tower import CoolingTower

K = 273.15


class OffDesignCycle:
    def __init__(self, dp, res, hs, x, eta_mc=0.85, eta_rc=0.85, eta_t=0.90, turbine_law="nozzle"):
        """Fix all hardware from the design point. dp: the design_point module; res, hs, x: dp.load_design()."""
        self.dp, self.res, self.hs, self.x0 = dp, res, hs, x
        st, m0 = res["states"], hs["m_co2"]
        self.st_des, self.m0 = st, m0
        self.p1_des = st[1]["p"]
        self.T_gas_des, self.m_gas_des = dp.EAF["T_in"], dp.EAF["m_gas"]
        self.dp_frac = {"LTR": dp.DP["LTR"], "HTR": dp.DP["HTR"]}
        # turbine and compressors
        self.turbine = trb.RadialTurbine(st[7]["T"], st[7]["p"], st[8]["p"], m0, eta_t, turbine_law)
        self.mc = cmp_.CompressorStage.from_design(st[1]["T"], st[1]["p"], st[2]["p"], (1.0 - x) * m0, eta_mc)
        self.rc = cmp_.Recompressor(cmp_.state_hp(st[10]["h"], st[10]["p"]), st[4]["p"], x * m0, st[4]["h"])
        # gas heater
        tr = ht.GasTransport(dp.EAF["X"])
        sh = ht.load_shares(dp.TABLES / "heater_resistance_shares.csv")
        self.heater = ht.SegmentedHeater(hs, st[6], st[7]["p"], m0, self.m_gas_des, sh, dp.GAS, transport=tr)
        # recuperators
        ref_r = pd.read_csv(SCRIPTS / "LTR & HTR" / "recuperator_reference_data.csv").set_index("parameter")["value"].astype(float)
        tab = rcp.CO2Table(dp.TABLES / "co2_recuperator_table.npz")
        D = dp.recuperator_design(res, hs, x)
        self.recup = {}
        for name, (ho, co) in (("HTR", (9, 6)), ("LTR", (10, 3))):
            s = pd.read_csv(dp.TABLES / f"{name}_resistance_shares.csv")
            r = rcp.SegmentedRecuperator(s, D[name], st[ho], st[co], ref_r, tab)
            r.attach_reynolds(s)
            self.recup[name] = r
        # water-cooled cooler and wet tower
        ref_c = pd.read_csv(SCRIPTS / "Cooler" / "cooler_reference_data.csv").set_index("parameter")["value"].astype(float)
        Dc = dp.water_cooler_design(res, hs, x)
        shc = pd.read_csv(dp.TABLES / "cooler_resistance_shares.csv")
        self.cooler = wc.WaterCooler(shc, Dc, ref_c, table_path=dp.TABLES / "co2_cooler_table.npz")
        self.cooler.tower = CoolingTower(ref_c["T_wb_des"], self.cooler.T_w_in_des - K - ref_c["dT_loop"],
                                         self.cooler.T_w_out_des - K - ref_c["dT_loop"], self.cooler.m_w_des,
                                         ref_c["LG_des"], ref_c["n_merkel"])
        self.T_wb_des = float(ref_c["T_wb_des"])
        self.u0 = np.array([m0, x, st[8]["p"], st[9]["p"], st[8]["h"], st[9]["h"], st[10]["h"]])
        self.last_u = self.u0.copy()
        self._tw_out = None
        self.n_eval = 0
        self.timers = {}

    def _t(self, key, t_start):
        self.timers[key] = self.timers.get(key, 0.0) + time.perf_counter() - t_start

    # ------------------------------------------------------------------ residuals
    def _eval(self, u, c):
        m, x, p8, p9, h8, h9, h10 = u
        if not (0.02 < x < 0.9) or m <= 1.0:
            raise ValueError("unknown out of range")
        mc_f, rc_f = (1.0 - x) * m, x * m
        self.n_eval += 1
        # cooler: state 10 -> state 1
        _t = time.perf_counter()
        cool = self.cooler.solve_coupled(c["T_wb"], c["r_cool"] * self.cooler.m_w_des, c["r_cool"] * self.cooler.m_air_des,
                                         h10, c["p1"], mc_f, tol=1e-4, max_iter=25, T_w_out0=self._tw_out)
        self._tw_out = cool["T_w_out"]
        self._t("cooler + tower", _t)
        _t = time.perf_counter()
        st1 = cmp_.state_hp(cool["h1"], c["p1"])
        # main compressor
        mcp = self.mc.operate(st1, mc_f, c["N_mc"] * self.mc.N_des)
        st2 = mcp["state_out"]
        self._t("main compressor", _t)
        _t = time.perf_counter()
        # LTR
        ltr = self.recup["LTR"].solve({"h": h9, "p": p9}, {"h": st2["h"], "p": st2["p"]}, m, mc_f)
        st3, st10 = ltr["cold_out"], ltr["hot_out"]
        self._t("LTR", _t)
        _t = time.perf_counter()
        # recompressor
        s10 = cmp_.state_hp(st10["h"], st10["p"])
        rcp_ = self.rc.operate(s10, rc_f, c["N_rc"] * self.rc.N_des)
        st4 = rcp_["state_out"]
        h5 = (1.0 - x) * st3["h"] + x * st4["h"]
        p5 = st3["p"]
        self._t("recompressor", _t)
        _t = time.perf_counter()
        # HTR
        htr = self.recup["HTR"].solve({"h": h8, "p": p8}, {"h": h5, "p": p5}, m, m)
        st6 = htr["cold_out"]
        self._t("HTR", _t)
        _t = time.perf_counter()
        # heater
        st6["T"] = cmp_.state_hp(st6["h"], st6["p"])["T"]
        hr = self.heater.solve(c["T_gas"], c["m_gas"], st6, m, c["bypass"])
        st7 = cmp_.state_Tp(hr["T7"], hr["p_out"])
        self._t("heater", _t)
        _t = time.perf_counter()
        # turbine
        tb = self.turbine.expand(st7["T"], st7["p"], p8)
        self._t("turbine", _t)
        res = np.array([(tb["m"] - m) / self.m0,
                        (st4["p"] - st3["p"]) / 1e4,
                        (tb["h_out"] - h8) / 1e4,
                        (htr["hot_out"]["h"] - h9) / 1e4,
                        (htr["hot_out"]["p"] - p9) / 1e3,
                        (st10["h"] - h10) / 1e4,
                        (st10["p"] - cool["p_in"]) / 1e3])
        info = {"m": m, "x": x, "cool": cool, "mcp": mcp, "rcp": rcp_, "ltr": ltr, "htr": htr, "hr": hr, "tb": tb,
                "st": {1: st1, 2: st2, 3: st3, 4: st4, 5: {"h": h5, "p": p5}, 6: st6, 7: st7, 8: {"h": h8, "p": p8},
                       9: {"h": h9, "p": p9}, 10: st10}, "residual": res}
        return res, info

    def solve(self, T_gas=None, m_gas=None, T_wb=None, N_mc=1.0, N_rc=1.0, r_cool=1.0, bypass=0.0, p1=None,
              guess=None, tol=1e-8, verbose=False):
        """Operating point. Speeds N_mc, N_rc are ratios of the design speed. Returns a dictionary of results (converged flag,
        residuals, states, flows, powers, heat input, stack temperature, component flags)."""
        c = {"T_gas": self.T_gas_des if T_gas is None else T_gas, "m_gas": self.m_gas_des if m_gas is None else m_gas,
             "T_wb": self.T_wb_des if T_wb is None else T_wb, "N_mc": N_mc, "N_rc": N_rc, "r_cool": r_cool,
             "bypass": bypass, "p1": self.p1_des if p1 is None else p1}
        u0 = self.u0
        z0 = (self.last_u if guess is None else np.asarray(guess, float)) / u0
        t0 = time.time()
        self.n_eval = 0

        def f(z):
            try:
                r, _ = self._eval(z * u0, c)
            except (ValueError, RuntimeError, ZeroDivisionError, OverflowError):
                return np.full(7, 1e3)
            if verbose:
                print(f"    eval {self.n_eval:3d}: max |residual| {np.abs(r).max():.2e}")
            return r
        sol = root(f, z0, method="hybr", tol=tol, options={"eps": 1e-5})
        u = sol.x * u0
        r, info = self._eval(u, c)
        info["converged"] = bool(np.abs(r).max() < 5e-4)
        info["max_residual"] = float(np.abs(r).max())
        info["u"], info["n_eval"], info["time"] = u, self.n_eval, time.time() - t0
        info["timers"] = dict(self.timers)
        if info["converged"]:
            self.last_u = u
        self._summarise(info, c)
        return info

    def _summarise(self, info, c):
        tb, hr, mcp, rcp_, cool = info["tb"], info["hr"], info["mcp"], info["rcp"], info["cool"]
        m, x = info["m"], info["x"]
        W_t, W_mc, W_rc = m * tb["w"], (1.0 - x) * m * mcp["w"], x * m * rcp_["w"]
        info["W_t"], info["W_mc"], info["W_rc"] = W_t, W_mc, W_rc
        info["W_net"] = W_t - W_mc - W_rc
        info["Q_heater"] = m * (info["st"][7]["h"] - info["st"][6]["h"])
        info["Q_gas"] = hr["Q"]
        info["eta_cycle"] = info["W_net"] / info["Q_heater"]
        info["W_aux"] = cool["P_aux"]
        info["W_net_after_aux"] = info["W_net"] - cool["P_aux"]
        info["T1"], info["T7"] = info["st"][1]["T"], info["st"][7]["T"]
        info["p2"], info["p7"], info["p8"] = info["st"][2]["p"], info["st"][7]["p"], info["st"][8]["p"]
        info["T_stack"] = hr["T_stack"]
        info["flags"] = {"MC surge": mcp["surge"], "MC outside map": mcp["outside"], "RC surge": rcp_["surge"],
                         "RC outside map": rcp_["outside"], "pressure above limit": bool(info["p2"] > self.dp.P_MAX * 1.0005),
                         "LTR Re outside": info["ltr"]["n_outside_Re"], "HTR Re outside": info["htr"]["n_outside_Re"],
                         "cooler temperature cross": bool(cool["cross"])}
