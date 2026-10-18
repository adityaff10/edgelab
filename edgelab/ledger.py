"""Append-only, hash-chained research ledger with an external checkpoint and a locked holdout.

What it guarantees, and what it does not:
- Every hypothesis must be registered (spec + pass/fail criteria) before results are recorded.
- Every evaluation recorded through this API counts as a trial, typed as new / parameter_change /
  rerun / robustness, so the multiple-testing penalty can use the honest number of attempts.
- Editing or deleting any entry breaks the hash chain. Deleting the LAST entry(ies), or rebuilding
  the whole chain, is only detectable against a checkpoint stored OUTSIDE the ledger file (for
  example committed to git, emailed, or printed in a report). `checkpoint()` writes one.
- It cannot see experiments run outside the ledger, and anyone with the raw data can bypass the
  holdout. It is a discipline tool and an audit trail, not a security boundary.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

TRIAL_TYPES = ("new", "parameter_change", "rerun", "robustness")


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
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def _append(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        entries = self.entries()
        prev = entries[-1]["hash"] if entries else "GENESIS"
        body = {"seq": len(entries), "kind": kind, "ts": datetime.now(timezone.utc).isoformat(),
                "payload": payload, "prev": prev}
        body["hash"] = hashlib.sha256(_canon(body).encode()).hexdigest()
        with self.path.open("a", newline="\n") as fh:
            fh.write(_canon(body) + "\n")
        return body

    def head(self) -> Dict[str, Any]:
        e = self.entries()
        return {"n_entries": len(e), "head_hash": e[-1]["hash"] if e else "GENESIS"}

    def verify_chain(self, checkpoint: Optional[Dict[str, Any]] = None) -> bool:
        """Internal consistency; with a checkpoint, also detects truncation and full rebuilds."""
        entries = self.entries()
        prev = "GENESIS"
        for i, e in enumerate(entries):
            body = {k: v for k, v in e.items() if k != "hash"}
            if e.get("seq") != i or e.get("prev") != prev:
                return False
            if hashlib.sha256(_canon(body).encode()).hexdigest() != e.get("hash"):
                return False
            prev = e["hash"]
        if checkpoint is not None:
            n = checkpoint["n_entries"]
            if len(entries) < n:
                return False                                   # entries removed after checkpoint
            at = entries[n - 1]["hash"] if n else "GENESIS"
            if at != checkpoint["head_hash"]:
                return False                                   # history rewritten
        return True

    def checkpoint(self, path: str | Path) -> Dict[str, Any]:
        """Write the current head to a separate file. Store that file somewhere the ledger isn't."""
        cp = {**self.head(), "written": datetime.now(timezone.utc).isoformat()}
        Path(path).write_text(json.dumps(cp, indent=2) + "\n", newline="\n")
        return cp

    # ---------- research API ----------
    def _find(self, kind: str, hid: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries() if e["kind"] == kind and e["payload"].get("id") == hid]

    def register(self, hid: str, spec: Dict[str, Any], criteria: Dict[str, Any],
                 provenance: Optional[Dict[str, Any]] = None, frozen: bool = False) -> Dict[str, Any]:
        """provenance: hashes of the pre-registration document, code version, data manifest, config."""
        if self._find("register", hid):
            raise LedgerError(f"hypothesis {hid} already registered; register a new id for a variant")
        if not criteria:
            raise LedgerError("pass/fail criteria are required before any evaluation")
        return self._append("register", {"id": hid, "spec": spec, "criteria": criteria,
                                         "provenance": provenance or {}, "frozen": frozen})

    def freeze(self, hid: str) -> Dict[str, Any]:
        if not self._find("register", hid):
            raise LedgerError(f"{hid} is not registered")
        return self._append("freeze", {"id": hid})

    def is_frozen(self, hid: str) -> bool:
        reg = self._find("register", hid)
        return bool(reg and reg[0]["payload"].get("frozen")) or bool(self._find("freeze", hid))

    def record_trial(self, hid: str, sample: str, result: Dict[str, Any], trial_type: str = "new",
                     provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if trial_type not in TRIAL_TYPES:
            raise LedgerError(f"trial_type must be one of {TRIAL_TYPES}")
        if not self._find("register", hid):
            raise LedgerError(f"cannot record results for unregistered hypothesis {hid}")
        if sample == "holdout" and not self._find("reveal", hid):
            raise LedgerError("holdout results require a logged reveal (use Holdout.reveal)")
        return self._append("trial", {"id": hid, "sample": sample, "trial_type": trial_type,
                                      "result": result, "provenance": provenance or {}})

    def n_trials(self, sample: Optional[str] = None, trial_type: Optional[str] = None) -> int:
        return sum(1 for e in self.entries() if e["kind"] == "trial"
                   and (sample is None or e["payload"]["sample"] == sample)
                   and (trial_type is None or e["payload"].get("trial_type") == trial_type))

    def trial_sharpes(self, sample: str = "development") -> List[float]:
        return [e["payload"]["result"]["sharpe"] for e in self.entries()
                if e["kind"] == "trial" and e["payload"]["sample"] == sample
                and "sharpe" in e["payload"]["result"]]


class Holdout:
    """Wraps untouched data. Released only to a frozen hypothesis, once. Software-level only:
    anyone with direct access to the raw files can bypass it."""

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
