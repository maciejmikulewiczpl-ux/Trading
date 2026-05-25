"""ntfy.sh push notifications for the ORB live runner.

Design rules (same as `live/status_ui.py`):
  - Pure decoration. Any failure here is logged and swallowed — trading must
    continue regardless of notification success/failure.
  - Silent no-op if the NTFY_TOPIC env var isn't set (so the trader works fine
    without any phone setup).
  - Short timeout (5s) — never block the main loop on a slow network.

Configuration:
  Set `NTFY_TOPIC` in `.env`, e.g.
      NTFY_TOPIC=orb-maciej-3f7a9c
  Then install the ntfy app on phone (iOS / Google Play) and subscribe to
  that exact topic. Topic acts as a shared secret — anyone who knows it can
  send to it, so use a random-ish string.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

log = logging.getLogger("orb_paper")

NTFY_BASE_URL = "https://ntfy.sh"
NTFY_TIMEOUT_SEC = 5


def notify(
    message: str,
    title: Optional[str] = None,
    priority: Optional[int] = None,
    tags: Optional[list[str]] = None,
) -> bool:
    """Send a push to the configured NTFY_TOPIC. Returns True if sent, False otherwise.

    priority: 1 (min) ... 5 (max). Default 3 by ntfy.
    tags: list of ntfy emoji names, see https://docs.ntfy.sh/emojis/
    """
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return False
    url = f"{NTFY_BASE_URL}/{topic}"
    headers = {}
    if title:
        headers["Title"] = title
    if priority is not None:
        headers["Priority"] = str(priority)
    if tags:
        headers["Tags"] = ",".join(tags)
    try:
        resp = requests.post(
            url,
            data=message.encode("utf-8"),
            headers=headers,
            timeout=NTFY_TIMEOUT_SEC,
        )
        if resp.status_code >= 400:
            log.warning(f"ntfy push failed: HTTP {resp.status_code} body={resp.text[:200]}")
            return False
        return True
    except Exception as e:
        log.warning(f"ntfy push failed: {e}")
        return False
