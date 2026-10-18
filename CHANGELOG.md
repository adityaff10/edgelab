# Changelog

## v1.0 — 2026-10-18

Fixes from an external review and from an independent reproduction on Windows.

- **Ledger:** external checkpoint (`checkpoint()`, `verify_chain(checkpoint)`) now detects deletion
  of the last entry and full chain rebuilds; previously undetected. Trials are typed
  (new / parameter_change / rerun / robustness). Registrations and trials carry provenance (hashes
  of the pre-registration, code, data manifest, git commit). Docs state the limits plainly.
- **Data:** `scripts/fetch_data.py` (cross-platform) replaces `fetch_data.sh`; clones with
  `core.autocrlf=false` at pinned commits and verifies `data_manifest.json`. New `content_sha256`
  normalises line endings: Windows' autocrlf had changed the raw-byte hash without changing data.
- **Options:** quote snapshots must carry timestamp and expiry; duplicates rejected; quotes from
  other times/expiries are ignored. New `structure_net_pnl_at_expiry` gives the complete trade
  result. Test now checks max net loss = wing × qty − credit + costs (v0.1 described payoff only).
- **E001:** post-hoc variance-form premium added (t = 0.06; 3.09 ex-COVID). c4 caveat added to
  results. `run_all` produces one consistent set of outputs (results, ledger, checkpoint, summary);
  v0.1's stale `ledger_trials` field is gone. Registered statistics are unchanged.
- **Validation:** no longer takes logs of non-positive prices (removed the test warning).
- v0.1 outputs are kept unchanged in `results/e001_v0.1/` for the audit trail.
- Docs: softened "premium very likely exists"; data-purchase checklist (≈22 months, bid/ask, licence).
