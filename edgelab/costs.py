"""Effective-dated cost and contract model for NSE index options.

Every number carries its source and whether it was verified against a source during this
build. Unverified entries are kept (so code runs) but are reported as warnings, never silently
trusted. Update this file, not the strategy code, when rules change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


@dataclass(frozen=True)
class Rate:
    start: date            # first date (trade date or expiry date, see field docs) the value applies
    end: Optional[date]    # last date it applies, None = still current
    value: float
    source: str
    verified: bool


def _pick(rates: List[Rate], d: date, name: str, warnings: Optional[list]) -> float:
    for r in rates:
        if r.start <= d and (r.end is None or d <= r.end):
            if not r.verified and warnings is not None:
                warnings.append(f"{name} on {d}: value {r.value} is UNVERIFIED ({r.source})")
            return r.value
    raise ValueError(f"No {name} defined for {d}; extend the schedule in edgelab/costs.py")


# STT on the SELL side of options, as a fraction of premium (trade date).
STT_OPTION_SELL = [
    Rate(date(2024, 10, 1), date(2026, 3, 31), 0.0010,
         "5paisa summary of Budget 2026 STT change (old rate 0.1%); start date of 0.1% not verified here",
         verified=False),
    Rate(date(2026, 4, 1), None, 0.0015,
         "https://www.5paisa.com/news/stt-hike-on-fo-to-take-effect-from-april-1-amid-rising-options-activity ; "
         "https://zerodha.com/charges", verified=True),
]

# STT on options bought and exercised (ITM at expiry), fraction of intrinsic value.
STT_OPTION_EXERCISE = [
    Rate(date(2024, 10, 1), date(2026, 3, 31), 0.00125, "5paisa (old rate 0.125%)", verified=False),
    Rate(date(2026, 4, 1), None, 0.0015, "https://zerodha.com/charges", verified=True),
]

# NSE exchange transaction charge on options, fraction of premium turnover (both sides).
NSE_TXN_OPTIONS = [
    Rate(date(2026, 1, 1), None, 0.0003553, "https://zerodha.com/charges (current; start date not verified)",
         verified=True),
    Rate(date(2020, 1, 1), date(2025, 12, 31), 0.0003553,
         "ASSUMED equal to current rate; historical values not verified", verified=False),
]

SEBI_FEE_PER_RUPEE = 10 / 1e7          # Rs 10 per crore of turnover, https://zerodha.com/charges
GST_RATE = 0.18                         # on brokerage + exchange + SEBI charges
STAMP_BUY = 0.00003                     # 0.003% on buy-side premium
BROKERAGE_PER_ORDER = 20.0              # Rs 20 flat per executed order (Zerodha); configurable

# Nifty lot size, keyed by CONTRACT EXPIRY date.
NIFTY_LOT = [
    Rate(date(2024, 11, 21), date(2025, 12, 30), 75,
         "SEBI Oct-2024 contract-size circular; exact value 75 not re-verified here", verified=False),
    Rate(date(2026, 1, 1), None, 65,
         "https://hdfcsky.com/news/nse-revises-market-lot-sizes-for-major-index-derivatives-effective-january-2026",
         verified=True),
]

# Weekday (0=Mon) of Nifty weekly expiry, keyed by expiry date.
NIFTY_EXPIRY_WEEKDAY = [
    Rate(date(2020, 1, 1), date(2025, 8, 31), 3, "Thursday expiries before Sept 2025", verified=True),
    Rate(date(2025, 9, 1), None, 1,
         "https://www.hdfcsec.com/blog/details/revision-of-nse-bse-expiry-day-of-derivatives-contracts",
         verified=True),
]


def stt_sell_rate(d: date, warnings: Optional[list] = None) -> float:
    return _pick(STT_OPTION_SELL, d, "STT(option sell)", warnings)


def stt_exercise_rate(d: date, warnings: Optional[list] = None) -> float:
    return _pick(STT_OPTION_EXERCISE, d, "STT(exercise)", warnings)


def nse_txn_rate(d: date, warnings: Optional[list] = None) -> float:
    return _pick(NSE_TXN_OPTIONS, d, "NSE txn(options)", warnings)


def nifty_lot_size(expiry: date, warnings: Optional[list] = None) -> int:
    return int(_pick(NIFTY_LOT, expiry, "Nifty lot size", warnings))


def nifty_expiry_weekday(expiry: date, warnings: Optional[list] = None) -> int:
    return int(_pick(NIFTY_EXPIRY_WEEKDAY, expiry, "Nifty expiry weekday", warnings))


@dataclass
class Fill:
    side: str          # "BUY" or "SELL"
    price: float       # premium per unit, rupees
    qty: int           # units (lots * lot size)


@dataclass
class CostBreakdown:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    gst: float = 0.0
    stamp: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.gst + self.stamp


def option_order_costs(fills: List[Fill], trade_date: date,
                       brokerage_per_order: float = BROKERAGE_PER_ORDER) -> CostBreakdown:
    """Statutory + broker costs for a list of option fills (one fill = one executed order)."""
    cb = CostBreakdown()
    stt = stt_sell_rate(trade_date, cb.warnings)
    txn = nse_txn_rate(trade_date, cb.warnings)
    for f in fills:
        if f.side not in ("BUY", "SELL"):
            raise ValueError(f"bad side {f.side}")
        if f.price < 0 or f.qty <= 0:
            raise ValueError("price must be >= 0 and qty > 0")
        turnover = f.price * f.qty
        cb.brokerage += brokerage_per_order
        cb.exchange += turnover * txn
        cb.sebi += turnover * SEBI_FEE_PER_RUPEE
        if f.side == "SELL":
            cb.stt += turnover * stt
        else:
            cb.stamp += turnover * STAMP_BUY
    cb.gst = GST_RATE * (cb.brokerage + cb.exchange + cb.sebi)
    return cb
