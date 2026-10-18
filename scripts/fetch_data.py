"""Fetch the public data used by E001 at PINNED commits, then verify content hashes.

Works the same on Windows, macOS and Linux:  python scripts/fetch_data.py --dest data/raw
Line-ending conversion is disabled on clone, and hashes are computed on normalised content,
so a Windows checkout verifies identically to a Linux one.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edgelab.data import verify_manifest  # noqa: E402

SOURCES = json.loads((ROOT / "data_manifest.json").read_text())


def run(*args: str) -> None:
    subprocess.run(list(args), check=True)


def main(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for name, src in SOURCES["repos"].items():
        target = dest / name
        if not target.exists():
            run("git", "clone", "-c", "core.autocrlf=false", src["url"], str(target))
        run("git", "-C", str(target), "-c", "advice.detachedHead=false", "checkout", "-q", src["commit"])
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        print(f"{name}: {head}{'' if head == src['commit'] else '  (MISMATCH)'}")
    problems = verify_manifest(dest, SOURCES["files"])
    if problems:
        print("DATA CHECK FAILED:\n  " + "\n  ".join(problems))
        return 1
    print(f"All {len(SOURCES['files'])} files match the manifest.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/raw", type=Path)
    sys.exit(main(ap.parse_args().dest))
