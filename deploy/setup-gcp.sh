#!/usr/bin/env bash
# ============================================================================
# SentiStack V2 — One-command GCP VM Setup
# ============================================================================
# Run this ONCE on a fresh Ubuntu 22.04 GCP Compute Engine VM:
#
#   bash setup-gcp.sh
#
# What it does:
#   1. Installs Docker + docker-compose-plugin
#   2. Installs git and helper tools
#   3. Clones your bot repo (or syncs if already present)
#   4. Installs the systemd service for auto-start on VM reboot
#   5. Prints next steps (add .env file, whitelist the static IP)
#
# Prerequisites (do these in GCP Console BEFORE running this script):
#   a) Create a VM:
#        gcloud compute instances create sentistack \
#          --machine-type=e2-small \
#          --zone=asia-south1-a \
#          --image-family=ubuntu-2204-lts \
#          --image-project=ubuntu-os-cloud \
#          --boot-disk-size=20GB \
#          --tags=sentistack
#
#   b) Reserve a STATIC external IP (crucial for Zerodha whitelist):
#        gcloud compute addresses create sentistack-ip --region=asia-south1
#        gcloud compute instances delete-access-config sentistack \
#          --access-config-name="External NAT" --zone=asia-south1-a
#        gcloud compute instances add-access-config sentistack \
#          --zone=asia-south1-a \
#          --address=$(gcloud compute addresses describe sentistack-ip \
#                       --region=asia-south1 --format='get(address)')
#
#   c) SSH into the VM:
#        gcloud compute ssh sentistack --zone=asia-south1-a
#
# ============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/YOUR_REPO.git}"
APP_DIR="${APP_DIR:-/opt/sentistack}"
SERVICE_FILE="/etc/systemd/system/sentistack.service"

echo "============================================================"
echo "  SentiStack V2 — GCP VM Setup"
echo "============================================================"

# ── 1. System updates ────────────────────────────────────────────────────────
echo "[1/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# ── 2. Install Docker ────────────────────────────────────────────────────────
echo "[2/6] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker "$USER"
    echo "  Docker installed. NOTE: You may need to log out and back in for"
    echo "  group membership to take effect (or use 'newgrp docker')."
else
    echo "  Docker already installed — skipping."
fi

# docker compose V2 (plugin, not standalone)
if ! docker compose version &> /dev/null 2>&1; then
    sudo apt-get install -y -qq docker-compose-plugin
fi
echo "  Docker Compose: $(docker compose version)"

# ── 3. Install helper tools ──────────────────────────────────────────────────
echo "[3/6] Installing git, jq, htop..."
sudo apt-get install -y -qq git jq htop

# ── 4. Clone / update repo ───────────────────────────────────────────────────
echo "[4/6] Setting up bot directory at $APP_DIR..."
sudo mkdir -p "$APP_DIR"
sudo chown "$USER":"$USER" "$APP_DIR"

if [ -d "$APP_DIR/.git" ]; then
    echo "  Repo already cloned — pulling latest..."
    git -C "$APP_DIR" pull
else
    echo "  Cloning repo from $REPO_URL ..."
    git clone "$REPO_URL" "$APP_DIR"
fi

# ── 5. Copy systemd service ──────────────────────────────────────────────────
echo "[5/6] Installing systemd service..."
sudo cp "$APP_DIR/deploy/sentistack.service" "$SERVICE_FILE"
# Substitute actual app dir into the service file
sudo sed -i "s|/opt/sentistack|$APP_DIR|g" "$SERVICE_FILE"
sudo systemctl daemon-reload
sudo systemctl enable sentistack
echo "  Service installed and enabled (starts on boot)."

# ── 6. Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!  Next steps:"
echo "============================================================"
echo ""
echo "  STEP 1 — Get the static IP of this VM and whitelist it in Zerodha:"
echo "    curl -s ifconfig.me"
echo "    → Add this IP to https://developers.kite.trade/ → your app → IP whitelist"
echo ""
echo "  STEP 2 — Create your .env file:"
echo "    cp $APP_DIR/.env.example $APP_DIR/.env"
echo "    nano $APP_DIR/.env"
echo "    (fill in all values — see comments in .env.example)"
echo ""
echo "  STEP 3 — Start the bot:"
echo "    sudo systemctl start sentistack"
echo "    sudo journalctl -u sentistack -f   # live logs"
echo ""
echo "  STEP 4 — Open Telegram and send /start to your bot."
echo "    The bot will send you its daily login URL every morning."
echo "    Use /token <request_token> to refresh daily without SSH."
echo ""
echo "  Useful commands:"
echo "    sudo systemctl status sentistack    # status"
echo "    sudo systemctl restart sentistack   # restart"
echo "    sudo systemctl stop sentistack      # stop"
echo "    docker compose -f $APP_DIR/docker-compose.yml logs -f bot  # live logs"
echo ""
