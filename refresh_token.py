#!/usr/bin/env python3
"""
refresh_token.py — SentiStack V2
=================================
Automated Kite access token refresh.

Usage:
    python3 refresh_token.py          # interactive browser flow
    python3 refresh_token.py --check  # just verify current token

What it does:
  1. Opens Zerodha login URL in your browser automatically
  2. You log in and paste the request_token from the redirect URL
  3. Exchanges it for a fresh access_token
  4. Saves it to ~/.env.sh AND .kite_token cache
  5. Prints the export command so you can source it immediately

Run this every morning before starting the bot, or set it up as a
9:00 AM cron job and paste the token from your phone.

Cron setup (runs at 8:55 AM IST = 3:25 AM UTC):
    crontab -e
    25 3 * * 1-5 cd ~/Trading_Bot && source venv/bin/activate && python3 refresh_token.py --auto >> ~/Trading_Bot/logs/token_refresh.log 2>&1
"""

import argparse
import json
import os
import re
import sys
import webbrowser
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

# Add bot directory to path
BOT_DIR = Path(__file__).parent
sys.path.insert(0, str(BOT_DIR))

try:
    from kiteconnect import KiteConnect
    from kiteconnect import exceptions as KiteExceptions
except ImportError:
    print("ERROR: kiteconnect not installed. Run: pip3 install kiteconnect")
    sys.exit(1)

ENV_FILE   = BOT_DIR / ".env.sh"
TOKEN_CACHE = BOT_DIR / ".kite_token"
IST = timedelta(hours=5, minutes=30)


def _ist_now() -> str:
    return datetime.now(timezone(IST)).strftime("%d %b %Y  %H:%M:%S IST")


def _load_env() -> dict:
    """Read key=value pairs from .env.sh."""
    env = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _update_env_file(new_token: str) -> None:
    """Update KITE_ACCESS_TOKEN in .env.sh in place."""
    if not ENV_FILE.exists():
        print(f"  WARNING: {ENV_FILE} not found — skipping .env.sh update.")
        print(f"  Manually add: export KITE_ACCESS_TOKEN=\"{new_token}\"")
        return

    content = ENV_FILE.read_text()
    pattern = r'(export\s+KITE_ACCESS_TOKEN\s*=\s*)["\']?[^"\'\n]*["\']?'
    replacement = f'export KITE_ACCESS_TOKEN="{new_token}"'

    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
    else:
        new_content = content.rstrip() + f'\nexport KITE_ACCESS_TOKEN="{new_token}"\n'

    ENV_FILE.write_text(new_content)
    ENV_FILE.chmod(0o600)
    print(f"  Updated {ENV_FILE}")


def _save_token_cache(token: str) -> None:
    payload = {"access_token": token, "generated_date": date.today().isoformat()}
    TOKEN_CACHE.write_text(json.dumps(payload, indent=2))
    TOKEN_CACHE.chmod(0o600)
    print(f"  Cached to {TOKEN_CACHE}")


def _check_cached_token(api_key: str) -> tuple:
    """Returns (token, is_valid_today) from cache."""
    if not TOKEN_CACHE.exists():
        return None, False
    try:
        data = json.loads(TOKEN_CACHE.read_text())
        token = data.get("access_token", "")
        gen_date = data.get("generated_date", "")
        is_today = gen_date == date.today().isoformat()
        return token, is_today
    except Exception:
        return None, False


def validate_token(api_key: str, token: str) -> bool:
    """Test token by calling kite.profile()."""
    k = KiteConnect(api_key=api_key)
    k.set_access_token(token)
    try:
        profile = k.profile()
        name = profile.get("user_name", "Unknown")
        uid  = profile.get("user_id", "")
        print(f"  Token valid — logged in as: {name} ({uid})")
        return True
    except KiteExceptions.TokenException:
        print("  Token INVALID or EXPIRED.")
        return False
    except Exception as exc:
        print(f"  Token check inconclusive (network?): {exc}")
        return True   # don't invalidate on network error


def run_oauth_flow(api_key: str, api_secret: str) -> str:
    """Interactive browser OAuth flow. Returns new access_token."""
    k = KiteConnect(api_key=api_key)
    login_url = k.login_url()

    print()
    print("=" * 60)
    print("  ZERODHA LOGIN")
    print("=" * 60)
    print()
    print("  Opening browser for Zerodha login...")
    print(f"  URL: {login_url}")
    print()
    print("  After login, you'll be redirected to a URL like:")
    print("    http://127.0.0.1/?request_token=XXXXXXXX&...")
    print()
    print("  Copy ONLY the request_token value and paste it below.")
    print()

    try:
        webbrowser.open(login_url)
        print("  Browser opened. If it didn't open, visit the URL above manually.")
    except Exception:
        print("  Could not open browser. Visit the URL above manually.")
    print()

    for attempt in range(1, 4):
        try:
            raw = input("  Paste request_token here -> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise RuntimeError("Login cancelled.")

        # Handle full URL being pasted
        if "request_token=" in raw:
            raw = raw.split("request_token=")[1].split("&")[0].strip()
            print(f"  (Extracted from URL: {raw[:8]}...)")

        if not raw:
            print(f"  Empty input (attempt {attempt}/3).")
            continue

        print("  Exchanging for access_token...")
        try:
            session = k.generate_session(raw, api_secret=api_secret)
            token   = session["access_token"]
            print("  Access token obtained!")
            return token
        except KiteExceptions.TokenException as exc:
            print(f"  Invalid request_token: {exc}")
            if attempt < 3:
                print(f"  Try again (attempt {attempt}/3).")
        except Exception as exc:
            print(f"  Error: {exc}")
            if attempt < 3:
                print(f"  Try again (attempt {attempt}/3).")

    raise RuntimeError("Failed after 3 attempts.")


def main():
    parser = argparse.ArgumentParser(description="Kite token refresh for SentiStack")
    parser.add_argument("--check",  action="store_true", help="Only verify current token")
    parser.add_argument("--auto",   action="store_true", help="Non-interactive (for cron)")
    parser.add_argument("--force",  action="store_true", help="Force refresh even if token is valid")
    args = parser.parse_args()

    print()
    print(f"  SentiStack Token Manager — {_ist_now()}")
    print()

    # Load credentials
    env = _load_env()
    api_key    = env.get("KITE_API_KEY") or os.environ.get("KITE_API_KEY", "")
    api_secret = env.get("KITE_API_SECRET") or os.environ.get("KITE_API_SECRET", "")
    curr_token = env.get("KITE_ACCESS_TOKEN") or os.environ.get("KITE_ACCESS_TOKEN", "")

    if not api_key or not api_secret:
        print("  ERROR: KITE_API_KEY or KITE_API_SECRET not found in .env.sh")
        sys.exit(1)

    print(f"  API Key: {api_key[:8]}...")

    # --check mode: just validate
    if args.check:
        print()
        print("  Checking current token...")
        if curr_token:
            valid = validate_token(api_key, curr_token)
            sys.exit(0 if valid else 1)
        else:
            print("  No KITE_ACCESS_TOKEN found.")
            sys.exit(1)

    # Check cache first (unless --force)
    if not args.force:
        cached_token, is_today = _check_cached_token(api_key)
        if is_today and cached_token:
            print("  Found today's cached token — validating...")
            if validate_token(api_key, cached_token):
                # Update .env.sh in case it's different
                if cached_token != curr_token:
                    print("  Updating .env.sh with cached token...")
                    _update_env_file(cached_token)
                print()
                print("  Token is valid. Run:")
                print(f"    source {ENV_FILE}")
                print("    python3 main.py")
                print()
                return
            print("  Cached token invalid — refreshing...")

    # Need new token
    if args.auto:
        print("  --auto mode: cannot run interactive browser flow.")
        print("  Please run without --auto to get a new token.")
        print("  Or set up a Telegram bot that triggers token refresh.")
        sys.exit(1)

    try:
        new_token = run_oauth_flow(api_key, api_secret)
    except RuntimeError as exc:
        print(f"\n  FAILED: {exc}")
        sys.exit(1)

    # Validate the new token
    print()
    print("  Validating new token...")
    if not validate_token(api_key, new_token):
        print("  ERROR: New token failed validation!")
        sys.exit(1)

    # Save everywhere
    print()
    print("  Saving token...")
    _update_env_file(new_token)
    _save_token_cache(new_token)

    print()
    print("=" * 60)
    print("  TOKEN REFRESHED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("  Now run:")
    print(f"    source {ENV_FILE}")
    print("    python3 main.py")
    print()
    print(f"  Or in one line:")
    print(f"    source {ENV_FILE} && python3 main.py")
    print()


if __name__ == "__main__":
    main()
