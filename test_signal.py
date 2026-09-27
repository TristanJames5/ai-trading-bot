from notifier import send_discord_alert
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# 1. Practice Standard Signal
signal_std = {
    "pair": "BTC-USD",
    "direction": "LONG",
    "entry": 64500.50,
    "sl": 63000.00,
    "tp": 68250.00,
    "grade": "A",
    "score": 85,
    "win_prob": 0.825,
    "session": "Crypto Weekend (Test)",
    "rr": 2.5,
    "time_utc": now,
    "engine": "v4.1 + v5 (Standard)"
}

# 2. Practice Sniper MAX Signal
signal_max = {
    "pair": "BTC-USD",
    "direction": "LONG",
    "entry": 64500.50,
    "sl": 63900.00,  # Much tighter stop loss
    "tp": 70000.00,  # Much higher take profit
    "grade": "A",
    "score": 85,
    "win_prob": 0.825,
    "session": "Crypto Weekend (Test)",
    "rr": 9.1,
    "time_utc": now,
    "engine": "v4.1 MAX + v5 MAX (Sniper)"
}

# 3. Practice PRO MAX Swing Signal
signal_swing = {
    "pair": "Gold",
    "direction": "SHORT",
    "entry": 2400.50,
    "sl": 2450.00,
    "tp": 2200.00,
    "grade": "A",
    "score": 90,
    "win_prob": 0.88,
    "session": "Daily Close",
    "rr": 4.0,
    "time_utc": now,
    "engine": "v4.1/v5 PRO MAX | Swing Trade (2-5 Days)"
}

print("Sending Practice Standard Signal...")
send_discord_alert(signal_std)

import time
time.sleep(2)

print("Sending Practice Sniper MAX Signal...")
send_discord_alert(signal_max)

time.sleep(2)

print("Sending Practice PRO MAX Swing Signal...")
send_discord_alert(signal_swing)

print("All practice signals sent to Discord!")
