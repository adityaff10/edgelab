"""Unit tests. Option-chain tests use a SYNTHETIC FIXTURE (labelled) to test arithmetic only."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from edgelab import costs
from edgelab.data import reconcile, validate_ohlc
from edgelab.ledger import Holdout, Ledger, LedgerError
from edgelab.options import bs_price, iron_fly, structure_entry, structure_payoff_at_expiry
from edgelab.stats import (deflated_sharpe, expected_max_sharpe, newey_west_tstat,
                           probabilistic_sharpe, stationary_bootstrap_ci)


# ---------------- costs ----------------
def test_stt_rate_changes_on_budget_date():
    assert costs.stt_sell_rate(date(2026, 4, 1)) == 0.0015
    w = []
    assert costs.stt_sell_rate(date(2026, 3, 31), w) == 0.0010
    assert w and "UNVERIFIED" in w[0]


def test_lot_size_and_expiry_weekday_schedules():
    assert costs.nifty_lot_size(date(2026, 1, 27)) == 65
    assert costs.nifty_expiry_weekday(date(2025, 9, 2)) == 1
    assert costs.nifty_expiry_weekday(date(2025, 8, 28)) == 3
    with pytest.raises(ValueError):
        costs.nifty_lot_size(date(2019, 1, 1))


def test_cost_breakdown_hand_computed():
    # one SELL of 65 units at Rs 100 on 2026-04-13 => turnover 6500
    cb = costs.option_order_costs([costs.Fill("SELL", 100.0, 65)], date(2026, 4, 13))
    assert cb.brokerage == 20
    assert cb.stt == pytest.approx(6500 * 0.0015)
    assert cb.exchange == pytest.approx(6500 * 0.0003553)
    assert cb.sebi == pytest.approx(6500 * 10 / 1e7)
    assert cb.gst == pytest.approx(0.18 * (20 + 6500 * 0.0003553 + 6500 * 1e-6))
    assert cb.stamp == 0
    buy = costs.option_order_costs([costs.Fill("BUY", 100.0, 65)], date(2026, 4, 13))
    assert buy.stt == 0 and buy.stamp == pytest.approx(6500 * 0.00003)


def test_cost_input_validation():
    with pytest.raises(ValueError):
        costs.option_order_costs([costs.Fill("HOLD", 1.0, 1)], date(2026, 4, 13))
    with pytest.raises(ValueError):
        costs.option_order_costs([costs.Fill("BUY", -1.0, 1)], date(2026, 4, 13))


# ---------------- ledger ----------------
def test_ledger_requires_registration_and_criteria(tmp_path):
    L = Ledger(tmp_path / "l.jsonl")
    with pytest.raises(LedgerError):
        L.record_trial("H1", "development", {"sharpe": 0.1})
    with pytest.raises(LedgerError):
        L.register("H1", {"rule": "x"}, {})
    L.register("H1", {"rule": "x"}, {"pass": "t>2"})
    with pytest.raises(LedgerError):
        L.register("H1", {"rule": "y"}, {"pass": "t>2"})
    L.record_trial("H1", "development", {"sharpe": 0.1})
    assert L.n_trials() == 1 and L.verify_chain()


def test_ledger_detects_tampering(tmp_path):
    p = tmp_path / "l.jsonl"
    L = Ledger(p)
    L.register("H1", {}, {"pass": "x"})
    L.record_trial("H1", "development", {"sharpe": -0.2})
    lines = p.read_text().splitlines()
    e = json.loads(lines[1])
    e["payload"]["result"]["sharpe"] = 0.9      # rewrite a bad result into a good one
    lines[1] = json.dumps(e, sort_keys=True, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    assert not L.verify_chain()


def test_ledger_detects_deleted_trial(tmp_path):
    p = tmp_path / "l.jsonl"
    L = Ledger(p)
    L.register("H1", {}, {"pass": "x"})
    L.record_trial("H1", "development", {"sharpe": -0.2})
    L.record_trial("H1", "development", {"sharpe": 0.5})
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], lines[2]]) + "\n")   # drop the failed trial
    assert not L.verify_chain()


def test_holdout_requires_freeze_and_reveals_once(tmp_path):
    L = Ledger(tmp_path / "l.jsonl")
    L.register("H1", {}, {"pass": "x"})
    h = Holdout(L, data=[1, 2, 3], label="2025-2026")
    with pytest.raises(LedgerError):
        h.reveal("H1")
    with pytest.raises(LedgerError):
        L.record_trial("H1", "holdout", {"sharpe": 1})
    L.freeze("H1")
    assert h.reveal("H1") == [1, 2, 3]
    with pytest.raises(LedgerError):
        h.reveal("H1")
    L.record_trial("H1", "holdout", {"sharpe": 0.2})
    assert L.verify_chain()


# ---------------- stats ----------------
def test_newey_west_matches_plain_t_without_lags():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1, 500)
    m, t = newey_west_tstat(x, lags=0)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert t == pytest.approx(plain, rel=1e-9)


def test_noise_rarely_significant():
    rng = np.random.default_rng(2)
    hits = sum(newey_west_tstat(rng.normal(0, 1, 250), 3)[1] > 2 for _ in range(400))
    assert hits / 400 < 0.06


def test_expected_max_sharpe_matches_monte_carlo():
    rng = np.random.default_rng(5)
    se = 1 / np.sqrt(11.5)                      # 800 noise strategies, 11.5 years, annual SR units
    mc = rng.standard_normal((20000, 800)).max(axis=1).mean() * se
    assert expected_max_sharpe(800, se ** 2) == pytest.approx(mc, rel=0.05)
    assert expected_max_sharpe(1, 0.01) == 0


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(3)
    x = rng.normal(0.05, 1, 1000)
    v = (1 / np.sqrt(1000)) ** 2
    psr = probabilistic_sharpe(x)
    assert deflated_sharpe(x, 1, v) == pytest.approx(psr)
    assert deflated_sharpe(x, 800, v) < deflated_sharpe(x, 10, v) < psr


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(4).normal(1.0, 1, 300)
    lo, hi = stationary_bootstrap_ci(x, n_boot=800)
    assert lo < x.mean() < hi


# ---------------- options ----------------
def test_put_call_parity():
    c = bs_price(24000, 24100, 0.05, 0.15, kind="C")
    p = bs_price(24000, 24100, 0.05, 0.15, kind="P")
    assert c - p == pytest.approx(24000 - 24100, abs=1e-6)


def _fixture_chain():
    # SYNTHETIC quotes for testing arithmetic only; not market data.
    rows = []
    for k, cb, ca, pb, pa in [(23500, 0, 0, 20.0, 21.0), (24000, 150.0, 151.0, 140.0, 141.0),
                              (24500, 18.0, 19.0, 0, 0)]:
        if ca:
            rows.append({"kind": "C", "strike": k, "bid": cb, "ask": ca})
        if pa:
            rows.append({"kind": "P", "strike": k, "bid": pb, "ask": pa})
    return pd.DataFrame(rows)


def test_iron_fly_entry_and_bounded_loss():
    legs = iron_fly(24000, 500)
    e = structure_entry(_fixture_chain(), legs, lot_size=65, trade_date=date(2026, 4, 13), model="cross")
    # cross: sell at bid (150+140), buy at ask (19+21) => 250 per unit
    assert e["net_credit"] == pytest.approx(250 * 65)
    worst = min(structure_payoff_at_expiry(legs, s, 65) for s in range(22000, 26001, 50))
    assert worst == pytest.approx(-500 * 65)          # loss capped at wing width
    mid = structure_entry(_fixture_chain(), legs, 65, date(2026, 4, 13), model="mid")
    assert mid["net_credit"] > e["net_credit"]         # mid fills are optimistic


def test_missing_strike_is_an_error_not_an_interpolation():
    with pytest.raises(KeyError):
        structure_entry(_fixture_chain(), iron_fly(24000, 1000), 65, date(2026, 4, 13))


# ---------------- data ----------------
def test_validation_flags_problems():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-02-20"])
    df = pd.DataFrame({"Open": 1, "High": [10, 10, 10, 5], "Low": [9, 9, 9, 6], "Close": [10, 10, -1, 5.5]},
                      index=idx)
    r = validate_ohlc(df, "bad")
    text = " ".join(r.issues)
    assert not r.ok and "duplicate" in text and "non-positive" in text and "gap" in text


def test_reconcile_finds_disagreement():
    a = pd.Series([100, 101, 102.0], index=pd.date_range("2024-01-01", periods=3))
    b = pd.Series([100, 101, 103.0], index=a.index)
    assert len(reconcile(a, b)) == 1
