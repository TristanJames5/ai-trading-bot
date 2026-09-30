"""
LOCAL DATABASE (SQLite)
=======================
Saves signals and trade journal entries to a local file (trading_data.db).
Captures ALL fields needed for 1-month statistical analysis.
"""

import sqlite3
from datetime import datetime, timezone, timedelta

DB_FILE = "trading_data.db"

# ─────────────────────────────────────────
# SCHEMA INIT + MIGRATION
# ─────────────────────────────────────────
def init_db():
    """Create tables if they don't exist. Migrate existing tables to add missing columns."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Create signals table with full schema
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            pair         TEXT,
            direction    TEXT,
            entry        REAL,
            sl           REAL,
            tp           REAL,
            grade        TEXT,
            score        INTEGER,
            session      TEXT,
            win_prob     REAL,
            rr           REAL,
            engine       TEXT,
            time_utc     TEXT,
            ltf_structure TEXT,
            fired_at     TEXT
        )
    ''')

    # Create trade_journal with full schema
    c.execute('''
        CREATE TABLE IF NOT EXISTS trade_journal (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id  INTEGER,
            result     TEXT,
            pnl_r      REAL,
            notes      TEXT,
            closed_at  TEXT,
            FOREIGN KEY(signal_id) REFERENCES signals(id)
        )
    ''')

    # --- MIGRATION: Add missing columns to existing DBs without data loss ---
    existing_cols = {row[1] for row in c.execute("PRAGMA table_info(signals)")}
    migration_cols = {
        "engine":        "TEXT DEFAULT ''",
        "time_utc":      "TEXT DEFAULT ''",
        "ltf_structure": "TEXT DEFAULT ''",
    }
    for col, col_def in migration_cols.items():
        if col not in existing_cols:
            c.execute(f"ALTER TABLE signals ADD COLUMN {col} {col_def}")
            print(f"  [DB MIGRATION] Added column '{col}' to signals table")

    conn.commit()
    conn.close()


# ─────────────────────────────────────────
# SAVE SIGNAL (captures everything)
# ─────────────────────────────────────────
def save_signal(signal: dict):
    """Save a fired signal with ALL fields to the local SQLite database."""
    try:
        init_db()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        c.execute('''
            INSERT INTO signals 
            (pair, direction, entry, sl, tp, grade, score, session, 
             win_prob, rr, engine, time_utc, ltf_structure, fired_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.get('pair',          ''),
            signal.get('direction',     ''),
            signal.get('entry',         0.0),
            signal.get('sl',            0.0),
            signal.get('tp',            0.0),
            signal.get('grade',         ''),
            signal.get('score',         0),
            signal.get('session',       ''),
            signal.get('win_prob',      0.0),
            signal.get('rr',            0.0),
            signal.get('engine',        ''),
            signal.get('time_utc',      ''),
            signal.get('ltf_structure', ''),
            datetime.now(timezone.utc).isoformat()
        ))

        last_id = c.lastrowid
        conn.commit()
        conn.close()
        print(f"  [DB] Signal saved (id={last_id}): {signal.get('pair')} {signal.get('direction')} | engine={signal.get('engine','?')}")
        return last_id

    except Exception as e:
        print(f"  [DB ERROR] Signal save failed: {e}")
        return None


# ─────────────────────────────────────────
# LOG TRADE RESULT
# ─────────────────────────────────────────
def log_trade_result(signal_id: int, result: str, pnl_r: float, notes: str = ""):
    """Log trade outcome to journal. Result = WIN | LOSS | BE (break-even)."""
    try:
        init_db()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        # Guard: don't double-log the same signal
        existing = c.execute(
            "SELECT id FROM trade_journal WHERE signal_id = ?", (signal_id,)
        ).fetchone()
        if existing:
            conn.close()
            return  # Already logged, skip

        c.execute('''
            INSERT INTO trade_journal 
            (signal_id, result, pnl_r, notes, closed_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            signal_id,
            result,
            pnl_r,
            notes,
            datetime.now(timezone.utc).isoformat()
        ))

        conn.commit()
        print(f"  [DB] Trade result logged: {result} | {pnl_r:+.2f}R (signal_id={signal_id})")
        conn.close()

    except Exception as e:
        print(f"  [DB ERROR] Journal save failed: {e}")


# ─────────────────────────────────────────
# GET ACTIVE (OPEN) SIGNALS
# ─────────────────────────────────────────
def get_active_signals():
    """Fetch signals from last 7 days that have not yet been closed in trade_journal."""
    try:
        init_db()
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        c.execute('''
            SELECT s.* FROM signals s
            LEFT JOIN trade_journal tj ON s.id = tj.signal_id
            WHERE tj.id IS NULL
            AND s.fired_at > ?
            ORDER BY s.fired_at ASC
        ''', (cutoff,))

        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    except Exception as e:
        print(f"  [DB ERROR] Failed to fetch active signals: {e}")
        return []


# ─────────────────────────────────────────
# ANALYTICS SUMMARY (for monthly review)
# ─────────────────────────────────────────
def get_performance_summary(days: int = 30):
    """
    Returns a performance summary dict for the last N days.
    Use this after 1 month to analyse your edge.
    """
    try:
        init_db()
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        rows = c.execute('''
            SELECT 
                s.pair, s.direction, s.grade, s.score, s.win_prob,
                s.session, s.engine, s.rr,
                tj.result, tj.pnl_r
            FROM signals s
            JOIN trade_journal tj ON s.id = tj.signal_id
            WHERE s.fired_at > ?
            ORDER BY s.fired_at ASC
        ''', (cutoff,)).fetchall()

        conn.close()

        if not rows:
            return {"error": "No closed trades in period"}

        total   = len(rows)
        wins    = sum(1 for r in rows if r['result'] == 'WIN')
        losses  = sum(1 for r in rows if r['result'] == 'LOSS')
        be      = sum(1 for r in rows if r['result'] == 'BE')
        total_r = sum(r['pnl_r'] for r in rows)
        win_rate = wins / total if total > 0 else 0

        # Per-pair breakdown
        pairs = {}
        for r in rows:
            p = r['pair']
            if p not in pairs:
                pairs[p] = {'wins': 0, 'losses': 0, 'total_r': 0.0, 'trades': 0}
            pairs[p]['trades']  += 1
            pairs[p]['total_r'] += r['pnl_r']
            if r['result'] == 'WIN':   pairs[p]['wins']   += 1
            if r['result'] == 'LOSS':  pairs[p]['losses'] += 1

        return {
            "period_days":  days,
            "total_trades": total,
            "wins":         wins,
            "losses":       losses,
            "break_evens":  be,
            "win_rate":     f"{win_rate:.1%}",
            "total_r":      round(total_r, 2),
            "per_pair":     pairs
        }

    except Exception as e:
        return {"error": str(e)}
