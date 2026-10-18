"""E001: volatility risk premium in Nifty (see E001_vrp_preregistration.md).

Usage: python experiments/e001_vrp.py --raw data/raw --out results/e001
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from edgelab import costs
from edgelab.data import load_ohlc, reconcile, sha256, validate_ohlc
from edgelab.ledger import Ledger
from edgelab.options import bs_price, iron_fly, structure_payoff_at_expiry
from edgelab.stats import concentration, newey_west_tstat, probabilistic_sharpe, stationary_bootstrap_ci

H = 21  # trading days
HID = "E001-VRP-NIFTY"

# Independent spot checks gathered from news sources during research (date, field, value, source).
SPOT_CHECKS = [
    ("2026-03-19", "nifty", 23002.15, "https://www.venturasecurities.com/?p=51444"),
    ("2026-03-30", "nifty", 22331.40, "https://www.angelone.in/news/market-updates/top-gainers-and-losers-on-march-30-2026-hindalco-leads-gains-bajaj-finance-tops-losers"),
    ("2026-03-19", "vix", 22.80, "https://www.venturasecurities.com/?p=51444"),
    ("2026-07-07", "vix", 11.64, "https://hdfcsky.com/news/india-vix-falls-1-52percent-to-11-64-as-easing-global-risks-push-fear-gauge-to-five-month-low"),
]


def load(raw: Path):
    f_n1 = raw / "FDRM_VIX/data/nifty50_daily_raw.csv"
    f_n2 = raw / "NIFTY_50_STOCK_DATA/NIFTY50_stock_history.csv"
    f_v = raw / "FDRM_VIX/data/india_vix_daily_raw.csv"
    n1, n2 = load_ohlc(f_n1), load_ohlc(f_n2)
    v = load_ohlc(f_v, rename={"Sum of Open": "Open", "Sum of High": "High",
                               "Sum of Low": "Low", "Sum of Close": "Close"})
    prov = {str(p): sha256(p) for p in (f_n1, f_n2, f_v)}
    return n1, n2, v, prov


def iron_fly_cost_fraction(spot: float, iv: float, trade_day: date, warn: list) -> dict:
    """Model-based: round-trip cost of a 1-lot iron fly as a fraction of its net credit.

    Prices legs with Black-Scholes at the given IV (no skew, no spread), 21 trading days to expiry,
    wings at 2% of spot. Adds a crossing cost of half of an ASSUMED bid/ask spread per leg.
    """
    t = H / 252
    lot = costs.nifty_lot_size(date(2026, 4, 28), warn)
    atm = round(spot / 50) * 50
    wing = round(spot * 0.02 / 50) * 50
    legs = iron_fly(atm, wing)
    px = {(l.kind, l.strike): bs_price(spot, l.strike, t, iv, kind=l.kind) for l in legs}
    credit = sum(px[(l.kind, l.strike)] * (1 if l.side == "SELL" else -1) for l in legs) * lot
    # entry + exit = 8 orders; exit priced at ~1/3 of entry premium (rough, assumed)
    fills = [costs.Fill(l.side, px[(l.kind, l.strike)], lot) for l in legs]
    exit_fills = [costs.Fill("BUY" if l.side == "SELL" else "SELL", px[(l.kind, l.strike)] / 3, lot)
                  for l in legs]
    stat = costs.option_order_costs(fills, trade_day).total + costs.option_order_costs(exit_fills, trade_day).total
    spread_cost = 8 * 0.5 * 1.0 * lot  # ASSUMED Rs 1.0 bid/ask per leg, pay half each time
    return {"credit": credit, "statutory_and_brokerage": stat, "spread_cost_assumed": spread_cost,
            "cost_fraction_of_credit": (stat + spread_cost) / credit, "lot": lot, "wing": wing}


def main(raw: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / "ledger.jsonl")
    criteria = {
        "c1": "mean vol spread > 0 and Newey-West(3) t > 2 on non-overlapping 21d windows",
        "c2": "median vol spread > 0",
        "c3": "mean vol spread > 0 in 2020-2022 and in 2023-end",
        "c4": "mean (IV-RV)/IV > model iron-fly round-trip cost fraction",
    }
    if not [e for e in ledger.entries() if e["kind"] == "register"]:
        ledger.register(HID, {"doc": "experiments/E001_vrp_preregistration.md", "horizon": H}, criteria,
                        frozen=True)

    n1, n2, vix, prov = load(raw)
    reports = [validate_ohlc(n1, "nifty_fdrm"), validate_ohlc(n2, "nifty_kalilur"),
               validate_ohlc(vix, "india_vix", max_abs_daily_move=0.6, lo=5, hi=95)]
    recon = reconcile(n1["Close"], n2["Close"])
    identical = bool(n1["Close"].reindex(n2.index).sub(n2["Close"]).abs().max() == 0)

    spot = []
    for d, field, val, src in SPOT_CHECKS:
        s = (n1 if field == "nifty" else vix)["Close"]
        got = float(s.get(pd.Timestamp(d), np.nan))
        spot.append({"date": d, "field": field, "expected": val, "got": got,
                     "match_within_0.1pct": bool(abs(got / val - 1) < 0.001) if not np.isnan(got) else None,
                     "source": src})

    df = pd.concat({"close": n1["Close"], "open": n1["Open"], "vix": vix["Close"]}, axis=1, join="inner").dropna()
    lr = np.log(df["close"]).diff()
    fwd_var = sum(lr.shift(-k) ** 2 for k in range(1, H + 1))
    df["rv"] = np.sqrt(fwd_var * 252 / H)
    df["iv"] = df["vix"] / 100
    df = df.dropna(subset=["rv", "iv"])
    df["spread"] = df["iv"] - df["rv"]
    df["vp"] = df["iv"] ** 2 - df["rv"] ** 2
    df["rel_edge"] = df["spread"] / df["iv"]

    nono = df.iloc[::H].copy()
    mean, t = newey_west_tstat(nono["spread"], lags=3)
    sub1 = nono.loc[:"2022-12-31", "spread"]
    sub2 = nono.loc["2023-01-01":, "spread"]

    # Model-priced short ATM straddle held 21 trading days (NOT a backtest), per unit of spot.
    sp = []
    for d, row in nono.iterrows():
        i = df.index.get_loc(d)
        j = i + H
        if j >= len(df.index):
            break
        # settle on the close H trading days later from the full Nifty series
        settle = n1["Close"].iloc[n1.index.get_loc(d) + H]
        s0, iv = row["close"], row["iv"]
        prem = bs_price(s0, s0, H / 252, iv, kind="C") + bs_price(s0, s0, H / 252, iv, kind="P")
        sp.append({"date": d, "pnl_pct_of_spot": (prem - abs(settle - s0)) / s0 * 100,
                   "premium_pct": prem / s0 * 100})
    sp = pd.DataFrame(sp).set_index("date")
    ex_mask = ~(((sp.index >= "2020-02-01") & (sp.index <= "2020-04-30")) |
                ((sp.index >= "2026-02-15") & (sp.index <= "2026-03-31")))

    warn: list = []
    cf = iron_fly_cost_fraction(float(df["close"].iloc[-1]), float(df["iv"].median()), date(2026, 4, 13), warn)

    # Overnight vs intraday share of variance.
    on = np.log(df["open"] / df["close"].shift(1))
    idy = np.log(df["close"] / df["open"])
    on_share = float((on ** 2).sum() / ((on ** 2).sum() + (idy ** 2).sum()))

    c1 = bool(mean > 0 and t > 2)
    c2 = bool(nono["spread"].median() > 0)
    c3 = bool(sub1.mean() > 0 and sub2.mean() > 0)
    c4 = bool(nono["rel_edge"].mean() > cf["cost_fraction_of_credit"])

    res = {
        "sample": {"start": str(df.index.min().date()), "end": str(df.index.max().date()),
                   "days": int(len(df)), "non_overlapping_windows": int(len(nono))},
        "vol_spread_mean_volpts": round(mean * 100, 2), "nw_t": round(t, 2),
        "vol_spread_median_volpts": round(float(nono["spread"].median()) * 100, 2),
        "share_windows_positive": round(float((nono["spread"] > 0).mean()), 3),
        "sub_2020_2022_mean_volpts": round(float(sub1.mean()) * 100, 2), "sub_2020_2022_n": int(len(sub1)),
        "sub_2023_end_mean_volpts": round(float(sub2.mean()) * 100, 2), "sub_2023_end_n": int(len(sub2)),
        "mean_iv": round(float(nono["iv"].mean()) * 100, 2), "mean_rv": round(float(nono["rv"].mean()) * 100, 2),
        "mean_rel_edge": round(float(nono["rel_edge"].mean()), 3),
        "median_rel_edge": round(float(nono["rel_edge"].median()), 3),
        "rel_edge_bootstrap_95ci": [round(v, 3) for v in stationary_bootstrap_ci(nono["rel_edge"], mean_block=3)],
        "all_days_mean_spread_volpts": round(float(df["spread"].mean()) * 100, 2),
        "iron_fly_cost_model": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in cf.items()},
        "iron_fly_cost_warnings": warn,
        "model_straddle": {
            "n": int(len(sp)), "mean_pnl_pct_spot": round(float(sp["pnl_pct_of_spot"].mean()), 3),
            "median_pnl_pct_spot": round(float(sp["pnl_pct_of_spot"].median()), 3),
            "win_rate": round(float((sp["pnl_pct_of_spot"] > 0).mean()), 3),
            "worst_pct_spot": round(float(sp["pnl_pct_of_spot"].min()), 2),
            "worst_date": str(sp["pnl_pct_of_spot"].idxmin().date()),
            "mean_premium_pct_spot": round(float(sp["premium_pct"].mean()), 2),
            "psr_vs_0": round(probabilistic_sharpe(sp["pnl_pct_of_spot"]), 3),
            "concentration_k3": {k: round(v, 2) for k, v in concentration(sp["pnl_pct_of_spot"], 3).items()},
            "mean_excluding_covid_and_mar2026": round(float(sp.loc[ex_mask, "pnl_pct_of_spot"].mean()), 3),
        },
        "overnight_share_of_variance": round(on_share, 3),
        "criteria": {"c1": c1, "c2": c2, "c3": c3, "c4": c4},
        "verdict": ("PASS" if all([c1, c2, c3, c4]) else
                    "PREMIUM EXISTS, NOT HARVESTABLE" if all([c1, c2, c3]) else "FAIL"),
    }
    ledger.record_trial(HID, "full", res)
    meta = {"provenance_sha256": prov, "validation": [r.__dict__ for r in reports],
            "nifty_sources_identical_on_common_dates": identical,
            "nifty_source_disagreements_gt_0.1pct": int(len(recon)), "spot_checks": spot,
            "ledger_chain_valid": ledger.verify_chain(), "ledger_trials": ledger.n_trials()}
    (out / "e001_results.json").write_text(json.dumps({"result": res, "meta": meta}, indent=2, default=str))
    sp.to_csv(out / "e001_model_straddle_windows.csv")
    nono[["iv", "rv", "spread"]].to_csv(out / "e001_windows.csv")
    print(json.dumps({"result": res, "meta": meta}, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--out", default="results/e001", type=Path)
    a = ap.parse_args()
    main(a.raw, a.out)
