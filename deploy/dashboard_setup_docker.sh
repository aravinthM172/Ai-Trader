#!/usr/bin/env bash
# Adds the trader dashboard to a VM whose ports 80/443 are already served by a Caddy CONTAINER
# (here: job-tracker-caddy-1 with ~/job-tracker/deploy/Caddyfile).  Same safe pattern as
# caddy_crewmill.sh: back up -> append one site block -> validate -> graceful reload; restore on failure.
# Nothing else on the VM is restarted or changed.  Safe to re-run (keeps secrets, skips existing block).
#
#   bash ~/dashboard_setup_docker.sh
set -euo pipefail

APP=/opt/trader-dashboard
ENVF=/etc/trader-dashboard.env
CADDY_C="${CADDY_CONTAINER:-job-tracker-caddy-1}"
CADDYFILE="${CADDYFILE:-$HOME/job-tracker/deploy/Caddyfile}"
NET="$(sudo docker inspect "$CADDY_C" --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}')"
IP="$(curl -s -4 --max-time 10 https://ifconfig.me || hostname -I | awk '{print $1}')"
HOST="${DASH_HOST:-${IP//./-}.sslip.io}"
NAME=trader-dashboard

echo "== trader dashboard (docker) :: caddy=$CADDY_C net=$NET host=$HOST"

# --- files -----------------------------------------------------------------------------
sudo mkdir -p "$APP/dashboard" "$APP/data/dashboard"
sudo cp ~/dashboard/cloud_app.py ~/dashboard/index.html "$APP/dashboard/"
sudo touch "$APP/dashboard/__init__.py"
sudo chown -R 10001:10001 "$APP/data"

# --- secrets (generated once) -------------------------------------------------------------
if ! sudo test -f "$ENVF"; then
  TOKEN="$(head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 40)"
  PASS="$(head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 20)"
  printf 'DASHBOARD_PUSH_TOKEN=%s\nDASHBOARD_USER=trader\nDASHBOARD_PASSWORD=%s\n' "$TOKEN" "$PASS" | sudo tee "$ENVF" >/dev/null
  sudo chmod 600 "$ENVF"
fi

# --- container (internal network only; no host ports) ----------------------------------------
sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
sudo docker run -d --name "$NAME" --restart unless-stopped --network "$NET" \
  --user 10001:10001 --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  --memory 128m --cpus 0.5 \
  --env-file "$ENVF" -e DASHBOARD_HOST=0.0.0.0 -e DASHBOARD_PORT=8080 -e DASHBOARD_DATA_DIR=/app/data/dashboard \
  -v "$APP/dashboard:/app/dashboard:ro" -v "$APP/data:/app/data" -w /app \
  python:3.12-alpine python -m dashboard.cloud_app >/dev/null
sleep 3
sudo docker exec "$NAME" python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/healthz').read().decode())"

# --- Caddy site block (append once, validate, reload) -------------------------------------------------
if grep -q "^$HOST {" "$CADDYFILE"; then
  echo "caddy block already present"
else
  BAK="$CADDYFILE.bak-$(date -u +%Y%m%d-%H%M)-before-trader-dashboard"
  cp "$CADDYFILE" "$BAK"
  cat >> "$CADDYFILE" <<BLOCK

# Trader dashboard (separate project: /opt/trader-dashboard) - added $(date -u +%Y-%m-%d)
$HOST {
	encode zstd gzip
	reverse_proxy $NAME:8080
	header Strict-Transport-Security "max-age=31536000"
}
BLOCK
  if ! sudo docker exec "$CADDY_C" caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/tmp/caddy-validate.log 2>&1; then
    echo "VALIDATION FAILED - restoring previous Caddyfile"; tail -5 /tmp/caddy-validate.log
    cp "$BAK" "$CADDYFILE"; exit 1
  fi
  sudo docker exec "$CADDY_C" caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile
  echo "caddy reloaded (other sites untouched)"
fi

echo
echo "=================================================================="
echo " Dashboard URL : https://$HOST   (certificate may take ~30 s on first visit)"
echo " Login         : $(sudo grep ^DASHBOARD_USER $ENVF | cut -d= -f2)  /  $(sudo grep ^DASHBOARD_PASSWORD $ENVF | cut -d= -f2)"
echo " PC .env lines :"
echo "   DASHBOARD_URL=https://$HOST"
echo "   DASHBOARD_PUSH_TOKEN=$(sudo grep ^DASHBOARD_PUSH_TOKEN $ENVF | cut -d= -f2)"
echo "=================================================================="
