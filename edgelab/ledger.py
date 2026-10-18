"""Append-only, hash-chained research ledger with a locked holdout.

Purpose: make self-deception expensive.
- Every hypothesis must be registered (spec + pass/fail criteria) before results are recorded.
- Every evaluation counts as a trial, so the multiple-testing penalty (deflated Sharpe) uses
  the honest number of attempts, including failed ones.
- The holdout period can be revealed only once per frozen hypothesis, and each reveal is logged.
- Entries are hash-chained, so silent edits or deletions are detectable.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class LedgerError(RuntimeError):
    pass


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    # ---------- low level ----------
    def entries(self) -> List[Dict[str, Any]]:
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def _append(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        entries = self.entries()
        prev = entries[-1]["hash"] if entries else "GENESIS"
        body = {"seq": len(entries), "kind": kind, "ts": datetime.now(timezone.utc).isoformat(),
                "payload": payload, "prev": prev}
        body["hash"] = hashlib.sha256(_canon(body).encode()).hexdigest()
        with self.path.open("a") as fh:
            fh.write(_canon(body) + "\n")
        return body

    def verify_chain(self) -> bool:
        prev = "GENESIS"
        for i, e in enumerate(self.entries()):
            h = e.get("hash")
            body = {k: v for k, v in e.items() if k != "hash"}
            if e.get("seq") != i or e.get("prev") != prev:
                return False
            if hashlib.sha256(_canon(body).encode()).hexdigest() != h:
                return False
            prev = h
        return True

    # ---------- research API ----------
    def _find(self, kind: str, hid: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries() if e["kind"] == kind and e["payload"].get("id") == hid]

    def register(self, hid: str, spec: Dict[str, Any], criteria: Dict[str, Any],
                 frozen: bool = False) -> Dict[str, Any]:
        if self._find("register", hid):
            raise LedgerError(f"hypothesis {hid} already registered; register a new id for a variant")
        if not criteria:
            raise LedgerError("pass/fail criteria are required before any evaluation")
        return self._append("register", {"id": hid, "spec": spec, "criteria": criteria, "frozen": frozen})

    def freeze(self, hid: str) -> Dict[str, Any]:
        if not self._find("register", hid):
            raise LedgerError(f"{hid} is not registered")
        return self._append("freeze", {"id": hid})

    def is_frozen(self, hid: str) -> bool:
        reg = self._find("register", hid)
        return bool(reg and reg[0]["payload"].get("frozen")) or bool(self._find("freeze", hid))

    def record_trial(self, hid: str, sample: str, result: Dict[str, Any]) -> Dict[str, Any]:
        if not self._find("register", hid):
            raise LedgerError(f"cannot record results for unregistered hypothesis {hid}")
        if sample == "holdout" and not self._find("reveal", hid):
            raise LedgerError("holdout results require a logged reveal (use Holdout.reveal)")
        return self._append("trial", {"id": hid, "sample": sample, "result": result})

    def n_trials(self, sample: Optional[str] = None) -> int:
        return sum(1 for e in self.entries()
                   if e["kind"] == "trial" and (sample is None or e["payload"]["sample"] == sample))

    def trial_sharpes(self, sample: str = "development") -> List[float]:
        return [e["payload"]["result"]["sharpe"] for e in self.entries()
                if e["kind"] == "trial" and e["payload"]["sample"] == sample
                and "sharpe" in e["payload"]["result"]]


class Holdout:
    """Wraps untouched data. Data is released only to a frozen hypothesis, once."""

    def __init__(self, ledger: Ledger, data: Any, label: str):
        self._ledger = ledger
        self._data = data
        self.label = label

    def reveal(self, hid: str) -> Any:
        if not self._ledger.is_frozen(hid):
            raise LedgerError(f"{hid} must be frozen before the holdout '{self.label}' is revealed")
        if self._ledger._find("reveal", hid):
            raise LedgerError(f"holdout already revealed to {hid}; re-testing on it would contaminate it")
        self._ledger._append("reveal", {"id": hid, "holdout": self.label})
        return self._data
