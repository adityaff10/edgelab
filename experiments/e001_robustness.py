"""Post-hoc robustness of E001. Logged as extra trials; cannot overturn the registered verdict.

Checks: (a) all 21 phase offsets of the non-overlapping sample, (b) Newey-West lag choice,
(c) overlapping daily sample with lag 21, (d) excluding COVID crash months.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab.data import load_ohlc
from edgelab.ledger import Ledger
from edgelab.stats import newey_west_tstat

H = 21


def main(raw: Path, out: Path):
    n = load_ohlc(raw / "FDRM_VIX/data/nifty50_daily_raw.csv")
    v = load_ohlc(raw / "FDRM_VIX/data/india_vix_daily_raw.csv",
                  rename={"Sum of Open": "Open", "Sum of High": "High", "Sum of Low": "Low", "Sum of Close": "Close"})
    df = pd.concat({"close": n["Close"], "vix": v["Close"]}, axis=1, join="inner").dropna()
    lr = np.log(df["close"]).diff()
    df["rv"] = np.sqrt(sum(lr.shift(-k) ** 2 for k in range(1, H + 1)) * 252 / H)
    df["spread"] = df["vix"] / 100 - df["rv"]
    df = df.dropna()

    phases = []
    for off in range(H):
        s = df["spread"].iloc[off::H]
        m, t = newey_west_tstat(s, 3)
        phases.append({"offset": off, "n": len(s), "mean_volpts": round(m * 100, 2), "t": round(t, 2)})
    tp = [p["t"] for p in phases]
    lags = {lag: round(newey_west_tstat(df["spread"].iloc[::H], lag)[1], 2) for lag in (0, 1, 3, 6)}
    m_all, t_all = newey_west_tstat(df["spread"], 21)
    ex = df[~(((df.index >= "2020-02-01") & (df.index <= "2020-04-30")))]
    m_ex, t_ex = newey_west_tstat(ex["spread"].iloc[::H], 3)
    # Variance form (IV^2 - RV^2). Added after an external review asked for it, i.e. after the
    # registered result was known; reported as post-hoc, never as part of the verdict.
    vp = ((df["vix"] / 100) ** 2 - df["rv"] ** 2)
    vp_n = vp.iloc[::H]
    m_vp, t_vp = newey_west_tstat(vp_n, 3)
    vp_ex = vp[~((vp.index >= "2020-02-01") & (vp.index <= "2020-04-30"))].iloc[::H]
    res = {
        "variance_premium_posthoc": {
            "mean_var_units": round(m_vp, 5), "t": round(t_vp, 2),
            "median_var_units": round(float(vp_n.median()), 5),
            "share_positive": round(float((vp_n > 0).mean()), 3),
            "t_excluding_feb_apr_2020": round(newey_west_tstat(vp_ex, 3)[1], 2),
            "reading": "Squaring makes the COVID window dominate; the variance premium is ~0 over the full sample.",
        },
        "phase_offsets_t": {"min": min(tp), "median": float(np.median(tp)), "max": max(tp),
                            "share_t_gt_2": round(float(np.mean(np.array(tp) > 2)), 3)},
        "phases": phases,
        "nw_lag_sensitivity_t": lags,
        "overlapping_daily_nw21": {"mean_volpts": round(m_all * 100, 2), "t": round(t_all, 2), "n": int(len(df))},
        "excluding_feb_apr_2020": {"mean_volpts": round(m_ex * 100, 2), "t": round(t_ex, 2)},
    }
    ledger = Ledger(out / "ledger.jsonl")
    ledger.record_trial("E001-VRP-NIFTY", "full", {k: v for k, v in res.items() if k != "phases"},
                        trial_type="robustness")
    (out / "e001_robustness.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "phases"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--out", default="results/e001", type=Path)
    a = ap.parse_args()
    main(a.raw, a.out)
