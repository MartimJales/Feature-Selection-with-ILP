#!/usr/bin/env python3
"""Quick Discord notification tester for pipeline troubleshooting."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests


def load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs from .env into process env without overriding existing vars."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send an immediate Discord test message")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".env",
        help="Path to .env file (default: repo/.env)",
    )
    parser.add_argument(
        "--webhook-url",
        default="",
        help="Discord webhook URL (overrides DISCORD_WEBHOOK_URL env)",
    )
    parser.add_argument(
        "--user-id",
        default="",
        help="Discord user ID to mention (overrides DISCORD_USER_ID env)",
    )
    parser.add_argument(
        "--message",
        default="🧪 Discord test message from Feature-Selection-with-ILP",
        help="Message body to send",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--no-mention",
        action="store_true",
        help="Do not mention the user even if user ID is set",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    load_env_file(args.env_file)

    webhook_url = (args.webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")).strip()
    user_id = (args.user_id or os.getenv("DISCORD_USER_ID", "")).strip()

    if not webhook_url:
        print("ERROR: DISCORD_WEBHOOK_URL not set (env or --webhook-url).")
        return 2

    content = args.message
    if user_id and not args.no_mention:
        content = f"<@{user_id}> {content}"

    print("Sending Discord test message...")
    print(f"- env file: {args.env_file}")
    print(f"- webhook configured: {'yes' if bool(webhook_url) else 'no'}")
    print(f"- user mention: {'yes' if bool(user_id and not args.no_mention) else 'no'}")

    try:
        response = requests.post(webhook_url, json={"content": content}, timeout=args.timeout)
    except Exception as exc:
        print(f"ERROR: request failed: {exc}")
        return 3

    print(f"HTTP status: {response.status_code}")
    response_text = (response.text or "").strip()
    if response_text:
        print(f"Response: {response_text[:500]}")

    if response.status_code in (200, 204):
        print("OK: message sent.")
        return 0

    print("ERROR: Discord rejected the request.")
    return 4


if __name__ == "__main__":
    sys.exit(main())
