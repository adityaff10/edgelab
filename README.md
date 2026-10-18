# edgelab

A small research lab for Indian index options whose main job is to **reject false
discoveries** before anyone trades them. It is not a trading bot; no live order path exists.

## Quick start (Windows PowerShell; on macOS/Linux use `source .venv/bin/activate`)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/fetch_data.py --dest data/raw          # pinned commits + content-hash check
python -m pytest -q                                   # 26 tests
python -m experiments.run_all --raw data/raw --out results/e001_mine
```

`run_all` runs the registered experiment, the post-hoc robustness checks, verifies the ledger,
writes `checkpoint.json` and `summary.json`. Compare your `summary.json` with
`results/e001/summary.json`; the statistics should match exactly.

Data is not redistributed (exchange-data licensing is restrictive). `data_manifest.json` pins the
source commits and lists LF-normalised content hashes, so Windows and Linux checkouts verify alike.

## How a hypothesis flows

1. **Write** the hypothesis and pass/fail rules first — `experiments/E001_vrp_preregistration.md`.
2. **Register** it in the ledger with provenance: hashes of the pre-registration, code, data manifest, and the git commit.
3. **Load and validate** data — `edgelab/data.py` (duplicates, gaps, OHLC consistency, large moves, manifest check).
4. **Compute** the measure — `experiments/e001_vrp.py`.
5. **Test** it — `edgelab/stats.py`; estimate economics — `edgelab/costs.py`, `edgelab/options.py`.
6. **Record** the trial, typed `new` / `parameter_change` / `rerun` / `robustness`. Failures count too.
7. **Verdict** against the pre-written rules — `e001_results.json`.
8. **Post-hoc checks**, logged as `robustness`, cannot overturn the verdict — `experiments/e001_robustness.py`.
9. **Checkpoint** the ledger head and store it outside the ledger file (commit it).

## Modules

| Module | Purpose |
| --- | --- |
| `edgelab/ledger.py` | Hash-chained ledger. Registration with criteria before results; typed trials; provenance per entry; once-only holdout for frozen hypotheses; external checkpoint detects edits, deletions (including the last entry) and full rebuilds. |
| `edgelab/stats.py` | Newey-West t, probabilistic and deflated Sharpe, expected max Sharpe of N noise strategies, stationary bootstrap, best/worst-k concentration. |
| `edgelab/costs.py` | Effective-dated NSE option costs and contract rules with source and `verified` flag; unverified values warn. |
| `edgelab/options.py` | Black-Scholes; iron fly; entry from a quote snapshot identified by **timestamp + expiry + kind + strike** (schema enforced, duplicates rejected, nothing interpolated); full net trade P&L at expiry. |
| `edgelab/data.py` | Loading, validation, reconciliation, raw and normalised content hashes, manifest verification. |

## What the ledger does and does not guarantee

- Detects: edited entries, deleted entries, deleted last entries and rebuilt chains — the last two
  **only against a checkpoint stored elsewhere** (e.g. committed to git).
- Does not detect: experiments run outside the ledger. Does not stop someone with the raw files
  from looking at the holdout. It is a discipline tool and audit trail, not a security boundary.

## E001 result

Question: does India VIX exceed the Nifty volatility realised over the next 21 trading days?
Sample 2 Jan 2020 – 10 Mar 2026, 73 non-overlapping windows. Independently reproduced on Windows
(Python 3.10) with identical numbers and, after fixing line-ending handling, identical data hashes.

| Measure | Value |
| --- | --- |
| Mean IV − RV | +1.89 vol points; Newey-West t = **1.96** (registered bar: 2.0) |
| Median IV − RV | +2.83 vol points; 76.7% of windows positive |
| 2020–22 / 2023–26 | +2.29 / +1.52 vol points |
| **Registered verdict** | **FAIL** (c1 missed; c2–c4 passed) |

Post-hoc (cannot overturn the verdict):

| Check | Result |
| --- | --- |
| Variance form, IV² − RV² | mean ≈ 0, **t = 0.06**; t = 3.09 excluding Feb–Apr 2020 |
| 21 window start offsets (vol form) | t from 1.83 to 2.58 |
| Overlapping days, lag 21 | t = 2.49 |
| Excluding Feb–Apr 2020 (vol form) | t = 4.57 |

**Reading.** The data is consistent with a volatility premium in normal periods, but the
registered test did not meet its bar, and in variance terms a single crash window erases six years
of it. That crash risk is what option sellers are paid for, so it must be counted. Criterion c4 is
a screening estimate only: it compares a straddle's relative edge with an iron fly's modelled costs
under assumed spreads, and does not show any tradable structure is profitable after costs.

**Decision.** No selling strategy on this evidence. The next test needs real option quotes.

## Before buying any options data

1. Get a **sample file** first and check it has: quote timestamps, expiry, strike, option type,
   **bid and ask** (not just OHLC candles), contract identifiers, full strike coverage.
2. Coverage needed for the planned E002: development Jan–Dec 2025 + holdout Jan–Oct 2026
   (≈22 months), all after the Nov 2024 expiry changes.
3. Confirm the licence permits private research use, and the total price.
4. Register E002 (exact rules, criteria) **before** the data arrives.

## Limitations

- Daily close-to-close data; nothing about intraday or 0DTE structures.
- VIX is a 30-day variance-style index, not the ATM IV a seller receives.
- The two Nifty files are byte-identical copies; four closes were spot-checked against news sources.
- Pre-2026 exchange charges and the start date of the 0.10% STT rate are unverified (flagged in code).
- Regulatory: SEBI's retail-algo framework requires black-box algo providers to register as research
  analysts (per NSE's FAQ on the framework); how this applies to any specific service needs
  professional confirmation. This project is personal research only.
