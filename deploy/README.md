# Deploy — MT5-free H1 forward test (Mode B, PC stays off)

Four steps. Everything after step 3 is automatic for ~4 weeks.

## 1. Create a free Oracle Cloud instance
Compute → Instances → Create:
- Image **Ubuntu 22.04/24.04**, shape **VM.Standard.A1.Flex** (Ampere ARM, Always-Free; 1 OCPU / 6 GB).
- Add your SSH public key. Note the **public IP**.

## 2. Build + upload the bundle (on the Windows PC, one time)
```powershell
cd D:\Downloads\Project\gold-ai-trader
powershell -ExecutionPolicy Bypass -File deploy\make_bundle.ps1
scp gold-ai-trader-bundle.tar.gz ubuntu@<PUBLIC_IP>:~/
```

## 3. Bootstrap the box (on the instance, one time)
```bash
ssh ubuntu@<PUBLIC_IP>
sudo apt update && sudo apt install -y python3-venv
mkdir -p gold-ai-trader && tar -xzf gold-ai-trader-bundle.tar.gz -C gold-ai-trader
cd gold-ai-trader
bash deploy/oracle_bootstrap.sh
```
This installs the venv (pandas/numpy/dotenv only), does a test fetch+replay,
and installs the daily cron. `order_send` is impossible here — `MetaTrader5`
is never imported; `LIVE_TRADING=false`.

## 4. Check in weekly
```bash
ssh ubuntu@<PUBLIC_IP> 'cd gold-ai-trader && bash deploy/status.sh'
```

Pre-registered rule (don't move the goalposts):
- < 20 forward trades → keep waiting.
- ≥ 15 forward trades and forward `expectancy_R` < **−0.10** → abort that asset.
- ≥ 20 forward trades, `expectancy_R` > 0, PF > 1 → continue (still not a green
  light: needs months + a regime change + a Mode-A confirm on the real Valetax
  feed + the `order_send` code, which does not exist).

## Stop early
```bash
ssh ubuntu@<PUBLIC_IP> 'cd gold-ai-trader && touch state/KILL_SWITCH && crontab -l | grep -v "#gold-ai-trader-fwd" | crontab -'
```

## Notes
- Feed: `tools/fetch_h1_public.py` pulls H1 from Binance (`BTCUSDT`, `PAXGUSDT`)
  with Yahoo fallbacks, on the box, no key. The in-sample prefix stays the real
  Valetax data from step 2; only bars after the cutoff (the forward test) are
  public, level-matched at the seam. Spread → Valetax broker floor.
- Full detail + Mode A (weekly PC export instead of public feed): `ORACLE_DEPLOY.md`.
