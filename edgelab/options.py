"""Option pricing and multi-leg structure evaluation against a quoted chain.

The chain evaluator needs REAL bid/ask quotes to produce meaningful results. The repository does
not ship real chain data (it is paid and licensed); tests use clearly labelled fixtures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import List

import pandas as pd
from scipy.stats import norm

from .costs import Fill, option_order_costs


def bs_price(spot: float, strike: float, t_years: float, vol: float, rate: float = 0.0,
             kind: str = "C") -> float:
    if t_years <= 0 or vol <= 0:
        intrinsic = max(0.0, spot - strike) if kind == "C" else max(0.0, strike - spot)
        return intrinsic
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol ** 2) * t_years) / (vol * math.sqrt(t_years))
    d2 = d1 - vol * math.sqrt(t_years)
    if kind == "C":
        return spot * norm.cdf(d1) - strike * math.exp(-rate * t_years) * norm.cdf(d2)
    return strike * math.exp(-rate * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)


@dataclass
class Leg:
    kind: str     # "C" or "P"
    strike: float
    side: str     # "BUY" or "SELL"
    lots: int = 1


def iron_fly(atm: float, wing: float, lots: int = 1) -> List[Leg]:
    """Short ATM straddle with long wings: max loss is bounded by the wing width (at expiry)."""
    return [Leg("C", atm, "SELL", lots), Leg("P", atm, "SELL", lots),
            Leg("C", atm + wing, "BUY", lots), Leg("P", atm - wing, "BUY", lots)]


def fill_price(quote: pd.Series, side: str, model: str = "cross", slip_ticks: int = 0,
               tick: float = 0.05) -> float:
    """Price we assume we get. 'mid' is optimistic; 'cross' pays the spread; slip adds ticks."""
    bid, ask = float(quote["bid"]), float(quote["ask"])
    if not (0 <= bid <= ask):
        raise ValueError(f"bad quote bid={bid} ask={ask}")
    if model == "mid":
        px = (bid + ask) / 2
    elif model == "cross":
        px = ask if side == "BUY" else bid
    else:
        raise ValueError(model)
    px += slip_ticks * tick if side == "BUY" else -slip_ticks * tick
    return max(px, 0.0)


def structure_entry(chain: pd.DataFrame, legs: List[Leg], lot_size: int, trade_date: date,
                    model: str = "cross", slip_ticks: int = 0) -> dict:
    """Net credit (positive = received) and costs for entering a structure from a quote snapshot.

    chain columns: kind, strike, bid, ask. Missing strikes raise, they are never interpolated.
    """
    fills, credit = [], 0.0
    for leg in legs:
        q = chain[(chain["kind"] == leg.kind) & (chain["strike"] == leg.strike)]
        if q.empty:
            raise KeyError(f"no quote for {leg.kind} {leg.strike}")
        px = fill_price(q.iloc[0], leg.side, model, slip_ticks)
        qty = leg.lots * lot_size
        fills.append(Fill(leg.side, px, qty))
        credit += px * qty if leg.side == "SELL" else -px * qty
    costs = option_order_costs(fills, trade_date)
    return {"net_credit": credit, "costs": costs.total, "cost_warnings": costs.warnings,
            "fills": fills}


def structure_payoff_at_expiry(legs: List[Leg], settle: float, lot_size: int) -> float:
    """Cash flow at expiry (negative = we pay) for the legs, excluding entry premium."""
    total = 0.0
    for leg in legs:
        intrinsic = max(0.0, settle - leg.strike) if leg.kind == "C" else max(0.0, leg.strike - settle)
        sign = 1 if leg.side == "BUY" else -1
        total += sign * intrinsic * leg.lots * lot_size
    return total
