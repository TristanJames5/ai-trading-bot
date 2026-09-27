"""
SUPABASE CLIENT
===============
Saves signals and trade journal entries to Supabase.
Setup:
  1. Create free account at supabase.com
  2. Create a new project
  3. Run the SQL from README.md to create tables
  4. Copy your project URL and anon key to .env
"""

import os
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def save_signal(signal: dict):
    """Save a fired signal to Supabase signals table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("  Supabase not configured — signal not saved to DB")
        print("  Set SUPABASE_URL and SUPABASE_KEY in .env to enable")
        return

    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)

        data = {
            "pair":      signal['pair'],
            "direction": signal['direction'],
            "entry":     signal['entry'],
            "sl":        signal['sl'],
            "tp":        signal['tp'],
            "grade":     signal['grade'],
            "score":     signal['score'],
            "win_prob":  signal['win_prob'],
            "session":   signal['session'],
            "rr":        signal['rr'],
            "fired_at":  datetime.utcnow().isoformat()
        }

        result = sb.table("signals").insert(data).execute()
        print(f"  Signal saved to Supabase (id: {result.data[0]['id'][:8]}...)")

    except Exception as e:
        print(f"  Supabase save failed: {e}")


def log_trade_result(signal_id: str, result: str, pnl_r: float, notes: str = ""):
    """
    Log trade outcome to journal.
    Call this manually after each trade closes.

    result: 'W' (win), 'L' (loss), 'BE' (break even)
    pnl_r:  actual R gained/lost (e.g. +2.5 or -1.0)
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return

    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)

        data = {
            "signal_id": signal_id,
            "result":    result,
            "pnl_r":     pnl_r,
            "notes":     notes,
            "closed_at": datetime.utcnow().isoformat()
        }

        sb.table("trade_journal").insert(data).execute()
        print(f"  Trade result logged: {result} | {pnl_r:+.2f}R")

    except Exception as e:
        print(f"  Journal save failed: {e}")
