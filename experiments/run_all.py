"""Run E001 end to end and leave one consistent set of outputs.

    python -m experiments.run_all --raw data/raw --out results/e001

Steps: registered experiment -> post-hoc robustness -> ledger verification -> checkpoint -> summary.
Store checkpoint.json somewhere the ledger isn't (commit it, email it) so truncation is detectable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgelab.ledger import Ledger
from experiments import e001_robustness, e001_vrp


def main(raw: Path, out: Path):
    e001_vrp.main(raw, out)
    e001_robustness.main(raw, out)
    ledger = Ledger(out / "ledger.jsonl")
    cp_path = out / "checkpoint.json"
    previous = json.loads(cp_path.read_text()) if cp_path.exists() else None
    ok = ledger.verify_chain(previous)
    cp = ledger.checkpoint(cp_path)
    res = json.loads((out / "e001_results.json").read_text())["result"]
    rob = json.loads((out / "e001_robustness.json").read_text())
    summary = {
        "hypothesis": "E001-VRP-NIFTY",
        "registered_verdict": res["verdict"],
        "criteria": res["criteria"],
        "vol_spread_mean_volpts": res["vol_spread_mean_volpts"], "nw_t": res["nw_t"],
        "variance_premium_posthoc": rob["variance_premium_posthoc"],
        "ledger": {"chain_valid_against_previous_checkpoint": ok,
                   "trials_total": ledger.n_trials(),
                   "trials_by_type": {t: ledger.n_trials(trial_type=t)
                                      for t in ("new", "rerun", "robustness", "parameter_change")},
                   "checkpoint": cp},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if not ok:
        raise SystemExit("LEDGER CHECK FAILED against the previous checkpoint")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--out", default="results/e001", type=Path)
    a = ap.parse_args()
    main(a.raw, a.out)
