#!/usr/bin/env bash
# One-time setup of the cloud dashboard on an Oracle Cloud Ubuntu VM (Always Free works).
# Installs: dashboard service (python3 stdlib only) + Caddy (automatic HTTPS via sslip.io).
#
#   scp -r dashboard deploy/dashboard_setup.sh ubuntu@<PUBLIC_IP>:~/
#   ssh ubuntu@<PUBLIC_IP> 'bash ~/dashboard_setup.sh'
#
# Prints the URL, the login and the push token to put in the PC's .env.
# Safe to re-run (keeps existing secrets).  The server never connects to MT5 and cannot trade.
set -euo pipefail

APP=/opt/trader-dashboard
ENVF=/etc/trader-dashboard.env
IP="$(curl -s -4 https://ifconfig.me || hostname -I | awk '{print $1}')"
HOST="${DASH_HOST:-${IP//./-}.sslip.io}"

echo "== trader dashboard setup =="
sudo apt-get update -qq
sudo apt-get install -y -qq python3 debian-keyring debian-archive-keyring apt-transport-https curl gnupg >/dev/null

# --- app files --------------------------------------------------------------------
sudo mkdir -p "$APP/dashboard" "$APP/data/dashboard"
sudo cp ~/dashboard/cloud_app.py ~/dashboard/index.html "$APP/dashboard/"
sudo touch "$APP/dashboard/__init__.py"
sudo useradd --system --no-create-home --shell /usr/sbin/nologin trader-dash 2>/dev/null || true
sudo chown -R trader-dash:trader-dash "$APP/data"

# --- secrets (generated once) -------------------------------------------------------------
if ! sudo test -f "$ENVF"; then
  TOKEN="$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 40)"
  PASS="$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 20)"
  sudo tee "$ENVF" >/dev/null <<EOF
DASHBOARD_PUSH_TOKEN=$TOKEN
DASHBOARD_USER=trader
DASHBOARD_PASSWORD=$PASS
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=8080
DASHBOARD_DATA_DIR=$APP/data/dashboard
EOF
  sudo chmod 600 "$ENVF"
fi

# --- systemd service ----------------------------------------------------------------------
sudo tee /etc/systemd/system/trader-dashboard.service >/dev/null <<EOF
[Unit]
Description=Trader dashboard (read-only snapshots from the trading PC)
After=network-online.target

[Service]
User=trader-dash
EnvironmentFile=$ENVF
WorkingDirectory=$APP
ExecStart=/usr/bin/python3 -m dashboard.cloud_app
Restart=always
RestartSec=5
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$APP/data
ProtectHome=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now trader-dashboard

# --- Caddy (HTTPS) -----------------------------------------------------------------------------
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq caddy >/dev/null
fi
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
$HOST {
    encode gzip
    reverse_proxy 127.0.0.1:8080
    header Strict-Transport-Security "max-age=31536000"
}
EOF
sudo systemctl reload caddy || sudo systemctl restart caddy

# --- Oracle Ubuntu images block 80/443 in iptables by default -------------------------------------------
for p in 80 443; do
  sudo iptables -C INPUT -p tcp --dport $p -m state --state NEW -j ACCEPT 2>/dev/null || \
    sudo iptables -I INPUT 6 -p tcp --dport $p -m state --state NEW -j ACCEPT
done
command -v netfilter-persistent >/dev/null && sudo netfilter-persistent save >/dev/null || true

echo
echo "=================================================================="
echo " Dashboard URL : https://$HOST"
echo " Login         : $(sudo grep ^DASHBOARD_USER $ENVF | cut -d= -f2)  /  $(sudo grep ^DASHBOARD_PASSWORD $ENVF | cut -d= -f2)"
echo
echo " Put these two lines in the TRADING PC's .env:"
echo "   DASHBOARD_URL=https://$HOST"
echo "   DASHBOARD_PUSH_TOKEN=$(sudo grep ^DASHBOARD_PUSH_TOKEN $ENVF | cut -d= -f2)"
echo
echo " Oracle console: VCN -> Security List -> add ingress TCP 80 and 443 from 0.0.0.0/0"
echo "=================================================================="
