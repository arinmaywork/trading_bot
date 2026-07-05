#!/usr/bin/env bash
# ============================================================================
# GCP VM: retire SentiStack (Docker stack) → install Wealth OS (lean, native)
# ============================================================================
# Run on the VM:  bash cleanup-gcp.sh
#
# What it does, in order:
#   1. Archives secrets + trade logs to ~/sentistack-archive-<date>.tgz  (nothing
#      is deleted before it is archived)
#   2. Stops & removes the sentistack systemd service
#   3. Tears down the Docker stack: containers, images, volumes, build cache
#   4. Optionally removes Docker itself (REMOVE_DOCKER=yes, default)
#   5. Migrates KITE_* and TELEGRAM_CHAT_ID into /opt/wealthos/.env.sh
#   6. Runs deploy/setup-oracle.sh (generic Ubuntu) to install Wealth OS
#   7. Cleans apt caches + old journals, prints disk before/after
#
# After it finishes: edit /opt/wealthos/.env.sh to add your NEW Wealth OS
# Telegram bot token, then re-run:  bash /opt/wealthos/deploy/setup-oracle.sh
# To restore data: gunzip your latest Telegram backup → /opt/wealthos/data/wealth.db
# ============================================================================
set -uo pipefail

REMOVE_DOCKER="${REMOVE_DOCKER:-yes}"
OLD=/opt/sentistack
ARCHIVE="$HOME/sentistack-archive-$(date +%Y%m%d).tgz"

echo "== Disk before =="
df -h / | tail -1

# ── 1. Archive everything worth keeping ─────────────────────────────
if [ -d "$OLD" ]; then
  echo "== Archiving secrets + logs to $ARCHIVE =="
  sudo tar czf "$ARCHIVE" \
    --ignore-failed-read \
    $(sudo find "$OLD" -maxdepth 3 \( -name ".env*" -o -name ".kite_token" \
        -o -name "*.csv" -o -name "wealth.db" \) 2>/dev/null) \
    2>/dev/null || true
  sudo chown "$USER" "$ARCHIVE" 2>/dev/null || true
  [ -f "$ARCHIVE" ] && echo "   archived: $(du -h "$ARCHIVE" | cut -f1)" \
                    || echo "   (nothing found to archive)"
fi

# ── 2. Stop old service ──────────────────────────────────────────────
echo "== Removing sentistack service =="
sudo systemctl disable --now sentistack 2>/dev/null || true
sudo rm -f /etc/systemd/system/sentistack.service
sudo systemctl daemon-reload

# ── 3. Docker teardown ───────────────────────────────────────────────
if command -v docker >/dev/null 2>&1; then
  echo "== Tearing down Docker stack =="
  (cd "$OLD" 2>/dev/null && sudo docker compose down -v --remove-orphans) || true
  sudo docker system prune -af --volumes || true
  if [ "$REMOVE_DOCKER" = "yes" ]; then
    echo "== Removing Docker entirely (Wealth OS doesn't need it) =="
    sudo apt-get purge -y -qq docker-ce docker-ce-cli containerd.io \
      docker-buildx-plugin docker-compose-plugin docker.io 2>/dev/null || true
    sudo rm -rf /var/lib/docker /var/lib/containerd
  fi
fi

# ── 4. Extract old secrets before deleting the tree ─────────────────
OLD_ENV=$(sudo find "$OLD" -maxdepth 3 -name ".env*" 2>/dev/null | head -1)
KITE_KEY=""; KITE_SECRET=""; CHAT_ID=""
if [ -n "$OLD_ENV" ]; then
  KITE_KEY=$(sudo grep -oP '(?<=KITE_API_KEY=)["\x27]?\K[^"\x27\s]+' "$OLD_ENV" | head -1 || true)
  KITE_SECRET=$(sudo grep -oP '(?<=KITE_API_SECRET=)["\x27]?\K[^"\x27\s]+' "$OLD_ENV" | head -1 || true)
  CHAT_ID=$(sudo grep -oP '(?<=TELEGRAM_CHAT_ID=)["\x27]?\K[^"\x27\s]+' "$OLD_ENV" | head -1 || true)
fi

echo "== Removing $OLD =="
sudo rm -rf "$OLD"

# ── 5. Pre-seed Wealth OS secrets ────────────────────────────────────
sudo mkdir -p /opt/wealthos
if [ ! -f /opt/wealthos/.env.sh ]; then
  sudo tee /opt/wealthos/.env.sh > /dev/null <<EOF
# Wealth OS secrets — migrated from SentiStack $(date +%F)
export TELEGRAM_BOT_TOKEN=""        # <<< PASTE YOUR NEW WEALTH OS BOT TOKEN
export TELEGRAM_CHAT_ID="${CHAT_ID}"
export KITE_API_KEY="${KITE_KEY}"
export KITE_API_SECRET="${KITE_SECRET}"
EOF
  sudo chmod 600 /opt/wealthos/.env.sh
  sudo chown "$USER" /opt/wealthos/.env.sh
fi

# ── 6. Install Wealth OS ─────────────────────────────────────────────
echo "== Installing Wealth OS =="
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if [ -f "$SCRIPT_DIR/setup-oracle.sh" ]; then
  bash "$SCRIPT_DIR/setup-oracle.sh"
else
  curl -fsSL https://raw.githubusercontent.com/arinmaywork/trading_bot/main/deploy/setup-oracle.sh | bash
fi

# ── 7. System cleanup ────────────────────────────────────────────────
echo "== apt + journal cleanup =="
sudo apt-get autoremove -y -qq || true
sudo apt-get clean
sudo journalctl --vacuum-size=100M >/dev/null 2>&1 || true
pip cache purge 2>/dev/null || true

echo "== Disk after =="
df -h / | tail -1
echo
echo "DONE. Next steps:"
echo "  1. nano /opt/wealthos/.env.sh   → paste the new bot token"
echo "  2. bash /opt/wealthos/deploy/setup-oracle.sh   (starts the service)"
echo "  3. Restore data: send latest wealth_*.db.gz from Telegram to the VM,"
echo "     gunzip it → /opt/wealthos/data/wealth.db, then: sudo systemctl restart wealthos"
echo "  4. Old secrets/logs archive: $ARCHIVE (download it, then delete)"
