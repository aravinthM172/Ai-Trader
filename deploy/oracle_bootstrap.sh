#!/usr/bin/env bash
# One-shot setup for the MT5-free H1 forward test on an Oracle Cloud (or any
# Linux) box.  Idempotent -- safe to re-run.  Never sends an order.
#
#   bash deploy/oracle_bootstrap.sh
#
# Prereqs on the box:  python3 + python3-venv  (sudo apt install -y python3-venv)
# Prereqs in this dir: .env with LIVE_TRADING=false, and the two cutoff datasets
#                      data/btcusd_vx_H1_dense.csv , data/xauusd_vx_H1.csv
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "== gold-ai-trader :: Oracle forward-test bootstrap =="
echo "   dir: $ROOT"

command -v python3 >/dev/null || { echo "!! install python3 first: sudo apt install -y python3-venv"; exit 1; }

[ -d venv ] || python3 -m venv venv
./venv/bin/pip -q install --upgrade pip
./venv/bin/pip -q install pandas numpy python-dotenv        # NO MetaTrader5, NO numba
mkdir -p data/export state reports logs

if [ -f .env ]; then
  grep -qE '^LIVE_TRADING=false' .env || { echo "!! .env must contain  LIVE_TRADING=false"; exit 1; }
else
  echo "LIVE_TRADING=false" > .env
  echo "   wrote .env (LIVE_TRADING=false)"
fi

for f in data/btcusd_vx_H1_dense.csv data/xauusd_vx_H1.csv; do
  [ -s "$f" ] || { echo "!! missing $f  -- scp it up once (it defines the forward-test cutoff)"; exit 1; }
done

echo
echo "-- test run: fetch public candles + replay both assets --"
./venv/bin/python -m tools.fetch_h1_public --asset ALL
./venv/bin/python run_btc.py --paper --timeframe H1             --from-csv data/export/btcusd_vx_H1_export.csv --paper-balance 1500
./venv/bin/python run_btc.py --paper --timeframe H1 --asset XAU --from-csv data/export/xauusd_vx_H1_export.csv --paper-balance 1500

echo
echo "-- install cron (idempotent) --"
TAG="#gold-ai-trader-fwd"
TMP="$(mktemp)"
crontab -l 2>/dev/null | grep -v "$TAG" > "$TMP" || true
cat >> "$TMP" <<EOF
20 6 * * * cd $ROOT && ./venv/bin/python -m tools.fetch_h1_public --asset ALL >> logs/fetch_cron.log 2>&1  $TAG
30 6 * * * cd $ROOT && ./venv/bin/python run_btc.py --paper --timeframe H1 --from-csv data/export/btcusd_vx_H1_export.csv --paper-balance 1500 >> logs/replay_cron.log 2>&1  $TAG
35 6 * * * cd $ROOT && ./venv/bin/python run_btc.py --paper --timeframe H1 --asset XAU --from-csv data/export/xauusd_vx_H1_export.csv --paper-balance 1500 >> logs/replay_cron.log 2>&1  $TAG
EOF
crontab "$TMP"
rm -f "$TMP"
crontab -l | grep "$TAG" || true

echo
echo "== done =="
echo "   results grow daily at 06:20-06:35 UTC."
echo "   read them:  bash deploy/status.sh"
echo "   stop it:    crontab -l | grep -v '$TAG' | crontab -   (and: touch state/KILL_SWITCH)"
