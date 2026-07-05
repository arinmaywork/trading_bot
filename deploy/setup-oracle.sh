#!/usr/bin/env bash
# Wealth OS — Oracle Cloud free tier setup (Ubuntu 22.04/24.04, x86 or ARM A1)
# Run as the default 'ubuntu' user:  bash setup-oracle.sh
set -euo pipefail

REPO="https://github.com/arinmaywork/trading_bot.git"
DIR=/opt/wealthos

echo "== 1. System packages =="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip git

echo "== 2. Clone / update repo =="
if [ -d "$DIR/.git" ]; then
  sudo git -C "$DIR" pull
else
  # dir may already exist (e.g. cleanup-gcp.sh pre-seeds .env.sh) — clone via
  # temp dir and merge, preserving any existing .env.sh / data/
  TMP=$(mktemp -d)
  git clone "$REPO" "$TMP"
  sudo mkdir -p "$DIR"
  sudo cp -a "$TMP"/. "$DIR"/
  rm -rf "$TMP"
fi
sudo chown -R "$USER":"$USER" "$DIR"

echo "== 3. Virtualenv (lean wealth_os deps only) =="
python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install -q --upgrade pip
"$DIR/venv/bin/pip" install -q -r "$DIR/wealth_os/requirements.txt"
# Kuvera PDF support: pdfplumber's pdfminer pin conflicts with casparser's,
# so install it in a second pass, then restore casparser's exact pdfminer.
"$DIR/venv/bin/pip" install -q "pdfplumber==0.11.9"
"$DIR/venv/bin/pip" install -q --no-deps --force-reinstall "pdfminer.six==20240706"

echo "== 4. Secrets =="
if [ ! -f "$DIR/.env.sh" ]; then
  cat <<'EOF' > "$DIR/.env.sh"
export TELEGRAM_BOT_TOKEN=""
export TELEGRAM_CHAT_ID=""
# T2 onwards:
export KITE_API_KEY=""
export KITE_API_SECRET=""
EOF
  chmod 600 "$DIR/.env.sh"
  echo ">>> EDIT $DIR/.env.sh with your tokens, then re-run this script. <<<"
  exit 0
fi

echo "== 5. systemd service =="
sudo tee /etc/systemd/system/wealthos.service > /dev/null <<EOF
[Unit]
Description=Wealth OS portfolio manager
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$DIR
ExecStart=/bin/bash -c 'source $DIR/.env.sh && exec $DIR/venv/bin/python -m wealth_os.main'
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "== 6. Watchdog (restarts service if heartbeat goes stale >10 min) =="
sudo tee /etc/systemd/system/wealthos-watchdog.service > /dev/null <<EOF
[Unit]
Description=Wealth OS heartbeat watchdog

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'f=$DIR/data/heartbeat; \
  if ! systemctl is-active --quiet wealthos; then systemctl restart wealthos; \
  elif [ ! -f "\$f" ] || [ \$(( \$(date +%s) - \$(stat -c %Y "\$f") )) -gt 600 ]; then \
    echo "heartbeat stale, restarting"; systemctl restart wealthos; fi'
EOF

sudo tee /etc/systemd/system/wealthos-watchdog.timer > /dev/null <<EOF
[Unit]
Description=Run Wealth OS watchdog every 5 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
EOF

echo "== 7. Journal size cap (log rotation) =="
sudo mkdir -p /etc/systemd/journald.conf.d
sudo tee /etc/systemd/journald.conf.d/wealthos.conf > /dev/null <<EOF
[Journal]
SystemMaxUse=200M
EOF
sudo systemctl restart systemd-journald

sudo systemctl daemon-reload
sudo systemctl enable --now wealthos
sudo systemctl enable --now wealthos-watchdog.timer
echo "== Done. Logs: sudo journalctl -u wealthos -f =="
