#!/usr/bin/env bash
# Fetches the public third-party copies of daily Nifty and India VIX used by E001.
# The data is NOT redistributed in this repository (licensing of exchange data is restrictive);
# it is fetched from its public sources and its SHA-256 is recorded in the experiment output.
set -euo pipefail
DEST="${1:-data/raw}"
mkdir -p "$DEST"
cd "$DEST"
[ -d FDRM_VIX ] || git clone --depth 1 https://github.com/Subham1709/FDRM_VIX
[ -d NIFTY_50_STOCK_DATA ] || git clone --depth 1 https://github.com/kalilurrahman/NIFTY_50_STOCK_DATA
echo "FDRM_VIX commit: $(git -C FDRM_VIX rev-parse HEAD)"
echo "NIFTY_50_STOCK_DATA commit: $(git -C NIFTY_50_STOCK_DATA rev-parse HEAD)"
