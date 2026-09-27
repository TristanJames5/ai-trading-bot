"""
EMAIL NOTIFIER
==============
Sends signal alerts via Gmail SMTP.
Setup: Create a Gmail App Password at:
  myaccount.google.com > Security > 2-Step Verification > App Passwords
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ─────────────────────────────────────────
# CONFIG — set these in your .env file
# ─────────────────────────────────────────
GMAIL_ADDRESS  = os.getenv("GMAIL_ADDRESS", "your@gmail.com")
GMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "your_app_password")
ALERT_TO       = os.getenv("ALERT_EMAIL", "your@gmail.com")   # Can be same or different


def send_signal_email(signal: dict):
    direction_arrow = "▲ LONG" if signal['direction'] == "LONG" else "▼ SHORT"
    grade_emoji = {"A": "★★★", "B": "★★☆", "C": "★☆☆"}.get(signal['grade'], "")

    subject = (
        f"[{signal['grade']}] {signal['pair']} {direction_arrow} "
        f"| {signal['session']}"
    )

    body = f"""
AI TRADING SIGNAL ALERT
{'='*45}

Pair        : {signal['pair']}
Direction   : {direction_arrow}
Grade       : {signal['grade']} {grade_emoji}
Session     : {signal['session']}
Time (UTC)  : {signal['time_utc']}

{'─'*45}
ENTRY       : {signal['entry']}
STOP LOSS   : {signal['sl']}
TAKE PROFIT : {signal['tp']}
R:R Ratio   : 1:{signal['rr']}
{'─'*45}

ENGINE SCORES:
  v4.1 Rules Score : {signal['score']}/100
  v5 ML Win Prob   : {signal['win_prob']*100:.1f}%
  Both engines AGREE → Signal fired

{'='*45}
⚠️  Decision support only. Always manage your risk.
    Only trade 1-2% risk per signal.
{'='*45}
"""

    try:
        msg = MIMEMultipart()
        msg['From']    = GMAIL_ADDRESS
        msg['To']      = ALERT_TO
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_PASSWORD)
            server.send_message(msg)

        print(f"  Email sent to {ALERT_TO}")

    except Exception as e:
        print(f"  Email failed: {e}")
        print(f"  Check GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env")
