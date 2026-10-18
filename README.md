# edgelab

A small research lab for Indian index options whose main job is to **reject false
discoveries** before anyone trades them. It is not a trading bot. Live execution does not exist
in this codebase.

## What it contains

| Module | Purpose |
| --- | --- |
| `edgelab/ledger.py` | Append-only, hash-chained ledger. Hypotheses must be registered with pass/fail criteria before results can be recorded; every evaluation counts as a trial; the holdout can be revealed once, only to a frozen hypothesis. Edits or deleted trials are detected. |
| `edgelab/stats.py` | Newey-West t-stat, probabilistic and deflated Sharpe (Bailey & López de Prado), expected max Sharpe of N noise strategies, stationary bootstrap, concentration (results without best/worst k). |
| `edgelab/costs.py` | Effective-dated NSE option costs and contract rules (STT 0.10% → 0.15% on 1 Apr 2026, exchange charges, SEBI fee, GST, stamp, Nifty lot 75 → 65, expiry Thursday → Tuesday). Each value carries its source and a `verified` flag; unverified values raise warnings. |
| `edgelab/options.py` | Black-Scholes, iron fly construction, entry from real bid/ask quotes (mid / cross / cross + slippage), expiry payoff. Missing strikes raise; nothing is interpolated. |
| `edgelab/data.py` | Loading, validation (duplicates, gaps, OHLC consistency, large moves), cross-source reconciliation, SHA-256 provenance. |
| `experiments/` | E001 (pre-registered) and its post-hoc robustness checks. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_data.sh data/raw      # clones public third-party copies of daily Nifty and India VIX
PYTHONPATH=. pytest -q                   # 18 tests
PYTHONPATH=. python experiments/e001_vrp.py --raw data/raw --out results/e001
PYTHONPATH=. python experiments/e001_robustness.py --raw data/raw --out results/e001
```

Data is not redistributed here. Exchange data licensing is restrictive; the fetch script pulls
public copies and the experiment records their SHA-256 hashes.

## E001 result (actually executed, 18 Oct 2026)

Question: does India VIX exceed the Nifty volatility realised over the following 21 trading days,
i.e. is there a volatility risk premium for option sellers to harvest? Criteria were written in
`experiments/E001_vrp_preregistration.md` before computing anything.

Sample: 2 Jan 2020 – 10 Mar 2026, 1,515 days, 73 non-overlapping 21-day windows.

| Measure | Value |
| --- | --- |
| Mean IV − RV | +1.89 vol points (IV 17.23%, RV 15.35%) |
| Newey-West t (lag 3) | 1.96 — **below the registered bar of 2.0** |
| Median IV − RV | +2.83 vol points; 76.7% of windows positive |
| 2020–2022 / 2023–2026 means | +2.29 / +1.52 vol points |
| Mean edge as share of premium | 11.5% (bootstrap 95% CI 0.4% – 19.7%) |
| Modelled 1-lot iron fly round-trip cost | ≈2% of credit (Rs 1 spread per leg assumed) |
| **Registered verdict** | **FAIL (criterion c1 missed; c2–c4 passed)** |

Post-hoc robustness (logged as extra trials; cannot overturn the verdict):

| Check | t-stat |
| --- | --- |
| 21 different window start offsets | 1.83 – 2.58 (median 2.25; 76% above 2) |
| Newey-West lag 0 / 1 / 3 / 6 | 1.95 / 1.90 / 1.96 / 2.19 |
| All overlapping days, lag 21 | 2.49 |
| Excluding Feb–Apr 2020 | 4.57 |

Model-priced short 21-day ATM straddle (Black-Scholes at VIX; **not a backtest**): mean +0.28% of
spot per window, median +1.07%, 68% wins, worst −22.8% (window starting 3 Mar 2020). Removing the
3 best windows cuts the total from 19.9 to 6.8; removing the 3 worst raises it to 54.8.

**Reading:** a ~2-vol-point premium very likely exists, but its statistical strength depends on
whether the COVID crash is counted. That crash is precisely the risk the premium pays for, so it
must be counted. Costs are not the binding constraint for monthly structures; tail risk is.

**Decision:** do not build a selling strategy on this evidence. The next gate needs real
historical option quotes (bid/ask, skew, intraday) to test a specific, registered structure.

## Limitations

- Daily close-to-close data; says nothing about intraday 0DTE straddles.
- VIX is a variance-swap style 30-day index, not the ATM IV a seller actually receives.
- The two "independent" Nifty files turned out to be byte-identical (same SHA-256), so the
  cross-source check is void. Four closes were spot-checked against news sources instead; all
  matched within 0.1%.
- Historical exchange charges and pre-2026 STT start dates are not verified (flagged in code).
- Straddle results are model-priced, not tradable fills.
