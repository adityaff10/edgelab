# E001 — Does Nifty carry a volatility risk premium large enough to matter?

Registered: 2026-10-18, before any VRP statistic was computed on this data.

## Why this experiment first

Every hedged option-selling strategy in the Reddit thread ultimately depends on one mechanism:
options are, on average, priced above the volatility that is later realised (the volatility
risk premium, VRP). If that premium is absent, small, or unstable in Nifty, the whole
"systematic option seller" direction has no mechanism, and building a selling bot is not justified.
If it is present, the direction survives this gate (it is necessary, not sufficient).

This can be tested with free, real data (daily Nifty closes and India VIX) without the paid
historical options data we do not have.

## Data

- India VIX daily close, 2020-01-02 to 2026-08-24 (third-party copy of ^INDIAVIX; source repo
  Subham1709/FDRM_VIX, which cites tanya459/india-vix-market-data-analysis).
- Nifty 50 daily OHLC from two independent third-party copies (Subham1709/FDRM_VIX and
  kalilurrahman/NIFTY_50_STOCK_DATA), reconciled against each other.
- Spot checks against closes reported in news sources found earlier (19 Mar 2026, 30 Mar 2026,
  7 Jul 2026).

## Definitions (fixed now)

- Implied vol at day t: IV_t = VIX_t / 100 (annualised, 30 calendar days).
- Realised vol: RV_t = sqrt(252 / 21 * sum of squared daily log returns over the NEXT 21 trading
  days t+1 .. t+21), close-to-close.
- Vol spread: S_t = IV_t - RV_t (vol points). Variance premium: VP_t = IV_t^2 - RV_t^2.
- Primary sample: non-overlapping 21-trading-day windows (every 21st day starting at the first
  valid day) to avoid overlap inflating significance.

## Primary hypothesis H1

Mean vol spread S over non-overlapping windows is positive.

## Pre-registered pass criteria (ALL must hold)

1. Mean S > 0 with Newey-West (lag 3) t-stat > 2.0 on the non-overlapping sample.
2. Median S > 0.
3. Mean S > 0 in BOTH sub-periods: 2020-01-01..2022-12-31 and 2023-01-01..end.
4. Economic test (model-based, labelled as such): the average edge as a fraction of an ATM
   straddle premium, (IV - RV)/IV, exceeds the estimated round-trip cost as a fraction of
   premium for a 1-lot hedged iron fly at current charges (costs computed by edgelab.costs).

## Fail / stop rule

If 1 or 3 fails: stop the option-selling direction; pivot to the fallback (tooling) only.
If 1-3 pass but 4 fails: the premium exists but is not harvestable by retail at current costs;
keep the lab, do not build a selling strategy.
If all pass: the direction survives this gate only; next gate requires real options chain data.

## Secondary (descriptive, no pass/fail)

- Distribution of a MODEL-PRICED short 21-day ATM straddle (Black-Scholes with VIX as IV,
  held to expiry). Clearly not a backtest: VIX is not the ATM IV and no real fills exist.
- Worst windows, share of total P&L from the best/worst windows, and the result with COVID
  (Feb-Apr 2020) and March 2026 included vs excluded.
- Fraction of realised variance that is overnight (close-to-open) vs intraday (open-to-close).

## What this cannot establish

- Profitability of any tradable strategy (no option quotes, spreads, fills, skew, or margin).
- Anything about intraday time-based straddles specifically.
- Durability beyond the sample.
