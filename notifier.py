"""
DISCORD NOTIFIER
=================
Sends signal alerts instantly to your Discord server for free.
"""

import os
import requests

def send_discord_alert(signal: dict):
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

    if not DISCORD_WEBHOOK_URL:
        print("  Discord not configured — skipping alert.")
        return

    color = 0x00FF00 if signal['direction'] == "LONG" else 0xFF0000
    grade_emoji = {"A": "⭐️⭐️⭐️", "B": "⭐️⭐️", "C": "⭐️"}.get(signal['grade'], "")

    embed = {
        "title": f"🤖 AI TRADING SIGNAL | {signal['pair']} {signal['direction']}",
        "color": color,
        "fields": [
            {"name": "Direction", "value": signal['direction'], "inline": True},
            {"name": "Grade", "value": f"{signal['grade']} {grade_emoji}", "inline": True},
            {"name": "Session", "value": signal['session'], "inline": True},
            {"name": "ENTRY", "value": f"```\n{signal['entry']}\n```", "inline": False},
            {"name": "STOP LOSS", "value": f"```\n{signal['sl']}\n```", "inline": True},
            {"name": "TAKE PROFIT", "value": f"```\n{signal['tp']}\n```", "inline": True},
            {"name": "Risk:Reward", "value": f"1:{signal['rr']}", "inline": False},
            {"name": "Engine Scores", "value": f"Rules Score: {signal['score']}/100\nML Win Prob: {signal['win_prob']*100:.1f}%", "inline": False}
        ],
        "footer": {"text": "Manage your risk. Only trade 1-2%."}
    }

    payload = {
        "embeds": [embed]
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code in [200, 204]:
            print("  [SUCCESS] Discord alert sent!")
        else:
            print(f"  [ERROR] Discord failed: {response.text}")
    except Exception as e:
        print(f"  [ERROR] Discord failed: {e}")
