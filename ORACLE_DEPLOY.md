# Oracle Cloud — H1 forward test (no always-on PC)

A free Oracle Cloud Linux box replays the strategy H1-by-H1. MetaTrader5 is
**not** installed there and is **never imported** by the replay path, so
`order_send()` is impossible. `LIVE_TRADING` stays `false`.

**Both** validated H1 candidates run — BTCUSD.vx and XAUUSD.vx — the same frozen
`momentum_rsi_mtf` strategy, selected by `--asset {BTC,XAU}`.

Pick the mode that matches your situation:

| | **Mode A — PC available weekly** | **Mode B — PC off the whole time** |
|---|---|---|
| Candle source | real Valetax feed, exported from MT5 on your PC | free public feed, fetched *on the cloud box* |
| BTC / XAU data | `tools.export_h1` → scp up weekly | `tools.fetch_h1_public` (Binance BTCUSDT / PAXGUSDT) |
| Your weekly effort | ~2 min (run + scp) | **zero** — cron does everything |
| Fidelity | exact broker OHLC + real per-bar spread | public OHLC, level-matched at the seam; spread set to the Valetax floor |

Both modes keep the **in-sample prefix = real Valetax data** (the dense CSVs you
scp once); only the bars **after the validation cutoff** — the actual forward
test — come from the chosen source. The replay is deterministic: each refresh
just extends the trade log. Trades split into **in-sample** and **FORWARD /
out-of-sample**; the FORWARD numbers are the test.

XAU physics is applied automatically: $100/lot value-per-unit, $0.31 broker
min-stop, spread floored at $0.30. BTC: $1/lot, $29.76 spread.

---

## 1. Create the instance (one time)

Oracle Cloud → Compute → Instances → **Create**:
- Image: **Canonical Ubuntu 22.04** (or 24.04)
- Shape: **VM.Standard.A1.Flex** (Ampere ARM) — *Always Free*, 1 OCPU / 6 GB is plenty.
  (The replay is pandas/numpy only, so ARM is fine. AMD `E2.1.Micro` also works.)
- Add your SSH public key, create.
- Note the **public IP**.

## 2. One-time setup on the instance

```bash
ssh ubuntu@<PUBLIC_IP>
sudo apt update && sudo apt install -y python3-venv git rsync
git clone <your repo>  gold-ai-trader        # or scp the project folder up
cd gold-ai-trader
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install pandas numpy python-dotenv        # NO MetaTrader5, NO numba needed
mkdir -p data/export state reports logs

# keep the validation cutoff files so "forward" is defined correctly (scp up once):
#   data/btcusd_vx_H1_dense.csv   (BTC cutoff = its last bar)
#   data/xauusd_vx_H1.csv         (XAU cutoff = its last bar)

cp .env .env 2>/dev/null || true            # ensure LIVE_TRADING=false is set
grep -E '^LIVE_TRADING' .env                # must print  LIVE_TRADING=false
```

`urllib` (Mode B's HTTP) is stdlib — nothing extra to install.

---

# Mode B — PC off the whole time (self-contained on the cloud box)

Everything runs on the instance. You do **nothing** for 4 weeks.

## 3B. One test run

```bash
cd $HOME/gold-ai-trader
./venv/bin/python -m tools.fetch_h1_public --asset ALL
./venv/bin/python run_btc.py --paper --timeframe H1            --from-csv data/export/btcusd_vx_H1_export.csv --paper-balance 1500
./venv/bin/python run_btc.py --paper --timeframe H1 --asset XAU --from-csv data/export/xauusd_vx_H1_export.csv --paper-balance 1500
```

`fetch_h1_public` pulls fresh H1 candles (BTC: Binance `BTCUSDT`; XAU: Binance
`PAXGUSDT`, Pax Gold ≈ spot gold; Yahoo fallbacks), shifts them by one constant
to meet the last real Valetax close at the seam, and writes the same
`data/export/*_H1_export.{csv,json}` the replay already reads. Spread has no
public source, so the column is set to the Valetax broker floor — the replay
labels this honestly (`spread_source: "public feed has no spread…"`).

## 4B. Cron (fetch, then replay — both assets)

```bash
crontab -e
```
add:
```
# 06:20 UTC daily: refresh public candles, then replay both assets
20 6 * * * cd $HOME/gold-ai-trader && ./venv/bin/python -m tools.fetch_h1_public --asset ALL >> logs/fetch_cron.log 2>&1
30 6 * * * cd $HOME/gold-ai-trader && ./venv/bin/python run_btc.py --paper --timeframe H1            --from-csv data/export/btcusd_vx_H1_export.csv --paper-balance 1500 >> logs/replay_cron.log 2>&1
35 6 * * * cd $HOME/gold-ai-trader && ./venv/bin/python run_btc.py --paper --timeframe H1 --asset XAU --from-csv data/export/xauusd_vx_H1_export.csv --paper-balance 1500 >> logs/replay_cron.log 2>&1
```

Then jump to **section 5**.

---

# Mode A — PC available weekly (exact Valetax feed)

## 3A. One test run

```bash
./venv/bin/python run_btc.py --paper --timeframe H1 \
    --from-csv data/export/btcusd_vx_H1_export.csv --paper-balance 1500
```

## 3A-weekly. On your Windows PC (MT5 running & logged in)

```powershell
cd D:\Downloads\Project\gold-ai-trader
.\venv\Scripts\python.exe -m tools.export_h1 --asset BTC --bars 24000
.\venv\Scripts\python.exe -m tools.export_h1 --asset XAU --bars 24000
scp data\export\btcusd_vx_H1_export.*  ubuntu@<PUBLIC_IP>:~/gold-ai-trader/data/export/
scp data\export\xauusd_vx_H1_export.*  ubuntu@<PUBLIC_IP>:~/gold-ai-trader/data/export/
```

(Or automate with Windows Task Scheduler + `pscp`/`scp`.)

## 4A. Cron on the instance (re-runs the replay after each new export)

```bash
crontab -e
```
add:
```
# every 6 h: re-replay both assets if a fresh export is present  (harmless if unchanged)
0 */6 * * * cd $HOME/gold-ai-trader && ./venv/bin/python run_btc.py --paper --timeframe H1 --from-csv data/export/btcusd_vx_H1_export.csv --paper-balance 1500 >> logs/replay_cron.log 2>&1
5 */6 * * * cd $HOME/gold-ai-trader && ./venv/bin/python run_btc.py --paper --timeframe H1 --asset XAU --from-csv data/export/xauusd_vx_H1_export.csv --paper-balance 1500 >> logs/replay_cron.log 2>&1
```

---

## 5. Read the result

```bash
for a in btc xau; do
  echo "== $a =="
  python3 -m json.tool < reports/${a}_h1_paper_status.json | grep -A12 forward_out_of_sample
done
column -s, -t reports/btc_paper_replay_trades.csv | less -S
column -s, -t reports/xau_paper_replay_trades.csv | less -S
```

`data_source` in each status file tells you which mode produced it
(`"MT5 export (Valetax)"` vs `"PUBLIC FEED (binance:…)"`).

The pre-registered pass/fail rule (do not move the goalposts):

| after the 4-week window | verdict |
|---|---|
| any bar-fill / sizing bug found | fix, restart the 4 weeks |
| ≥ 15 forward trades and forward `expectancy_R` < **−0.10** | **abort** — the edge did not survive |
| ≥ 20 forward trades, `expectancy_R` > 0, PF > 1 | **continue** — still not a green light; needs months + a regime change + the `order_send` code |

## Safety on the cloud box

- `MetaTrader5` is never imported by the replay path (`test_replay_import_chain_is_mt5_free` proves it), so `order_send()` cannot be called.
- `LIVE_TRADING=false`; `state/KILL_SWITCH` (touch the file) aborts the replay.
- No broker credentials are needed on the cloud box — only candle CSVs and (Mode B) outbound HTTPS to Binance/Yahoo.
- `--paper-balance 1500` is a *hypothetical* sizing number; your real $100 account is never referenced there.

## Mode B fidelity caveat

The public feed matches broker OHLC closely but not exactly (different venue,
no real spread). It is good enough for a **go / no-go** forward test — if the
edge only shows up on one specific feed it was never real. Before any live
decision, re-run at least once in **Mode A** (one PC session) to confirm the
forward numbers hold on the true Valetax feed.
