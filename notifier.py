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
            {"name": "Engine", "value": signal.get('engine', 'v4.1 + v5 (Standard)'), "inline": True},
            {"name": "ENTRY", "value": f"```\n{signal['entry']}\n```", "inline": False},
            {"name": "STOP LOSS", "value": f"```\n{signal['sl']}\n```", "inline": True},
            {"name": "TAKE PROFIT", "value": f"```\n{signal['tp']}\n```", "inline": True},
            {"name": "Risk:Reward", "value": f"1:{signal['rr']}", "inline": False},
            {"name": "Scores", "value": f"Rules: {signal['score']}/100 | ML: {signal['win_prob']*100:.1f}%", "inline": False}
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

def send_discord_result(signal: dict, result: str, pnl_r: float):
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not DISCORD_WEBHOOK_URL: return

    color = 0x00FF00 if result == "WIN" else 0xFF0000
    emoji = "✅" if result == "WIN" else "❌"

    embed = {
        "title": f"{emoji} TRADE CLOSED | {signal['pair']} {signal['direction']}",
        "color": color,
        "fields": [
            {"name": "Result", "value": result, "inline": True},
            {"name": "Net Profit", "value": f"{pnl_r:+.2f} R", "inline": True},
        ]
    }

    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=10)
    except:
        pass
