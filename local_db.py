"""
LOCAL DATABASE (SQLite)
=======================
Saves signals and trade journal entries to a local file (trading_data.db).
Zero setup, zero limits, runs offline, 100% free.
"""

import sqlite3
from datetime import datetime

DB_FILE = "trading_data.db"

def init_db():
    """Create tables if they don't exist yet."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            direction TEXT,
            entry REAL,
            sl REAL,
            tp REAL,
            grade TEXT,
            score INTEGER,
            session TEXT,
            win_prob REAL,
            rr REAL,
            fired_at TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            result TEXT,
            pnl_r REAL,
            notes TEXT,
            closed_at TEXT,
            FOREIGN KEY(signal_id) REFERENCES signals(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_signal(signal: dict):
    """Save a fired signal to the local SQLite database."""
    try:
        init_db()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO signals 
            (pair, direction, entry, sl, tp, grade, score, session, win_prob, rr, fired_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['pair'],
            signal['direction'],
            signal['entry'],
            signal['sl'],
            signal['tp'],
            signal['grade'],
            signal['score'],
            signal['session'],
            signal['win_prob'],
            signal['rr'],
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        print(f"  Signal saved to local database (trading_data.db)")
        conn.close()

    except Exception as e:
        print(f"  Local DB save failed: {e}")

def log_trade_result(signal_id: int, result: str, pnl_r: float, notes: str = ""):
    """Log trade outcome to journal."""
    try:
        init_db()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO trade_journal 
            (signal_id, result, pnl_r, notes, closed_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            signal_id,
            result,
            pnl_r,
            notes,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        print(f"  Trade result logged to local DB: {result} | {pnl_r:+.2f}R")
        conn.close()

    except Exception as e:
        print(f"  Journal save failed: {e}")
