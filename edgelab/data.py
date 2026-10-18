"""Loading and validating daily index data. Validation failures are returned, not hidden."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    name: str
    rows: int = 0
    start: str = ""
    end: str = ""
    issues: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.startswith("ERROR") for i in self.issues)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ohlc(path: str | Path, date_col: str = "Date", rename: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if rename:
        df = df.rename(columns=rename)
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
    df = df.set_index(date_col).sort_index()
    df.index.name = "Date"
    cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    return df[cols].astype(float)


def validate_ohlc(df: pd.DataFrame, name: str, max_abs_daily_move: float = 0.15,
                  lo: float | None = None, hi: float | None = None) -> ValidationReport:
    r = ValidationReport(name, len(df), str(df.index.min().date()), str(df.index.max().date()))
    if df.index.has_duplicates:
        r.issues.append(f"ERROR duplicate dates: {int(df.index.duplicated().sum())}")
    if not df.index.is_monotonic_increasing:
        r.issues.append("ERROR dates not sorted")
    nan = int(df["Close"].isna().sum())
    if nan:
        r.issues.append(f"WARN {nan} missing closes")
    c = df["Close"].dropna()
    if (c <= 0).any():
        r.issues.append("ERROR non-positive closes")
    if {"High", "Low"} <= set(df.columns):
        bad = ((df["High"] < df["Low"]) | (df["Close"] > df["High"] * 1.0001) |
               (df["Close"] < df["Low"] * 0.9999)).sum()
        if bad:
            r.issues.append(f"WARN {int(bad)} rows where Close is outside [Low, High] or High < Low")
    ret = np.log(c).diff().dropna()
    big = ret[ret.abs() > max_abs_daily_move]
    for d, v in big.items():
        r.issues.append(f"WARN large move {v:+.2%} on {d.date()} (check against an official source)")
    gaps = c.index.to_series().diff().dt.days
    for d, g in gaps[gaps > 7].items():
        r.issues.append(f"WARN gap of {int(g)} calendar days ending {d.date()}")
    if lo is not None and (c < lo).any():
        r.issues.append(f"WARN values below plausible floor {lo}")
    if hi is not None and (c > hi).any():
        r.issues.append(f"WARN values above plausible ceiling {hi}")
    return r


def reconcile(a: pd.Series, b: pd.Series, tol: float = 0.001) -> pd.DataFrame:
    """Rows where two sources disagree by more than tol (relative), on common dates."""
    j = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
    j["rel_diff"] = (j["a"] / j["b"] - 1).abs()
    return j[j["rel_diff"] > tol]
