#!/usr/bin/env bash
# Print the current forward-test standing for BTC and XAU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./venv/bin/python
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
[ -n "$PY" ] || { echo "no python found"; exit 1; }

for a in btc xau; do
  f="reports/${a}_h1_paper_status.json"
  echo "==================== ${a^^} ===================="
  if [ ! -s "$f" ]; then echo " (no run yet)"; continue; fi
  "$PY" - "$f" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
i, fw = s["in_sample"], s["forward_out_of_sample"]
print(" data source :", s.get("data_source"))
print(" spread      :", s.get("spread_source"))
print(" cutoff      :", s["validation_cutoff_utc"])
print(" candles     : %s  (%s -> %s)" % (s["candles"], *s["candle_span"]))
print(" in-sample   : %s trades  PF %s  expR %s" % (i.get("trades"), i.get("profit_factor"), i.get("expectancy_R")))
print(" FORWARD     : %s trades  net $%s  PF %s  expR %s  maxDD $%s  maxConsecL %s" % (
    fw.get("trades"), fw.get("net_pl_usd"), fw.get("profit_factor"),
    fw.get("expectancy_R"), fw.get("max_drawdown_usd"), fw.get("max_consecutive_losses")))
n, e = fw.get("trades") or 0, fw.get("expectancy_R")
if n < 15:
    print("  -> verdict: too few forward trades yet (need >= 20)")
elif e is not None and e < -0.10:
    print("  -> verdict: ABORT this asset -- forward expectancy < -0.10 R")
elif n >= 20 and e is not None and e > 0 and (fw.get("profit_factor") or 0) > 1:
    print("  -> verdict: CONTINUE (not a green light -- needs months + a Mode-A confirm + order_send)")
else:
    print("  -> verdict: inconclusive, keep collecting")
PY
done
echo
echo "live_trading_enabled: $(grep -o '"live_trading_enabled":[^,]*' reports/btc_h1_paper_status.json 2>/dev/null || echo n/a)"
