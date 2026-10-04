# Cloud dashboard

Open it from anywhere (phone or laptop). Password-protected, HTTPS, read-only: it cannot trade.

```
Trading PC (MT5 + bots)                  Oracle Cloud VM (Always Free)            You
 dashboard/publisher.py  ──HTTPS push──▶  dashboard/cloud_app.py + Caddy   ◀───  browser
 every 60 s, outbound only                stores latest snapshot
```

## What it shows
- **Status chips:** PC online / silent, trader, watchdog, MT5, LIVE vs dry-run, algo trading, kill switch,
  next entry window. Each chip pairs a colour with an icon and a word.
- **Challenge tiles:** equity; phase and profit vs target (with progress bar); room left today; total room.
- **Price chart (H1)** for each symbol: every trade marked (▲ buy, ▼ sell, ● exit with reason and R),
  open-position entry/SL/TP lines, and the **next-trade plan** (dotted entry/SL/TP if the signal fires).
- **Next trade panel:** BUY/SELL/no signal from the last completed bar, each condition met or not yet
  (e.g. "RSI ≤ 40, now 40.7"), and the exact entry / 2 ATR stop / 3 ATR target.
- Equity curve, open positions, trades table, news blackouts, live-vs-backtest edge monitor, latest alerts.

## Set up the cloud (once, about 15 min)
1. Oracle Cloud → create an **Ubuntu 22.04/24.04** instance (Always Free `VM.Standard.A1.Flex`), add your SSH key,
   note the public IP. In its **VCN → Security List**, add ingress rules for TCP **80** and **443** from `0.0.0.0/0`.
2. From the project folder on the PC:
   ```powershell
   scp -r dashboard deploy/dashboard_setup.sh ubuntu@<PUBLIC_IP>:~/
   ssh ubuntu@<PUBLIC_IP> "bash ~/dashboard_setup.sh"
   ```
   It prints the **URL** (`https://<ip-with-dashes>.sslip.io`), the **login**, and two lines for the PC's `.env`.
3. Add those two lines to the PC's `.env`:
   ```
   DASHBOARD_URL=https://...sslip.io
   DASHBOARD_PUSH_TOKEN=...
   ```
4. On the PC, test once with `venv\Scripts\python -m dashboard.publisher --once` (it should print `pushed -> HTTP 200`),
   then run **`start_dashboard_publisher.bat`** alongside `start_live_multi.bat` and `start_watchdog.bat`.
5. Open the URL and log in. Save it to your phone's home screen.

## Safety
- The PC only sends data out; no ports are opened on the PC; the cloud has no route to MT5.
- Pushes need the secret token; viewing needs the login. Both live in `/etc/trader-dashboard.env` (mode 600) on the VM.
- If the PC stops, the header shows **"PC silent N min"** in red; the watchdog's Telegram alerts cover it too.
- Update the page later: re-copy `dashboard/` and run `sudo cp ~/dashboard/* /opt/trader-dashboard/dashboard/ && sudo systemctl restart trader-dashboard`.
