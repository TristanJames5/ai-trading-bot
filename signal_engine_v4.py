"""
FINAL SIGNAL ENGINE — v4
=========================
Architecture:
  Layer 1: Structural Bias (Daily EMA trend)
  Layer 2: Zone Quality (Premium/Discount + OB/FVG)
  Layer 3: Session Kill Zone Trigger (London/NY only)
  Layer 4: Per-Pair Optimized Parameters

Price Action Concepts:
  - Asia Range Sweep detection (manipulation before real move)
  - Fair Value Gap (FVG) identification
  - Institutional Order Block (OB)
  - Liquidity grabs (PDH/PDL sweeps)
  - Premium/Discount zones (50% Fibonacci)

This file is the PRODUCTION signal engine.
Run it to get live signals for any configured pair.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ─────────────────────────────────────────
# PER-PAIR OPTIMIZED CONFIGS
# (Will be updated once optimizer.py completes)
# ─────────────────────────────────────────
PAIR_CONFIGS = {
    "US30": {
        "ticker":       "YM=F",
        "sl_atr":       1.2,
        "tp_atr":       3.0,
        "score_thresh": 55,
        "rsi_low":      40,
        "rsi_high":     60,
        "sma_dist_max": 2.5,
        "sessions":     ["london", "ny"],
        "pip_div":      1,        # price divisor for pips (indices = raw pts)
    },
    "NAS100": {
        "ticker":       "NQ=F",
        "sl_atr":       1.2,
        "tp_atr":       3.0,
        "score_thresh": 55,
        "rsi_low":      40,
        "rsi_high":     60,
        "sma_dist_max": 2.5,
        "sessions":     ["london", "ny"],
        "pip_div":      1,
    },
    "Gold": {
        "ticker":       "GC=F",
        "sl_atr":       1.5,      # Gold needs wider stops
        "tp_atr":       3.0,
        "score_thresh": 55,
        "rsi_low":      35,
        "rsi_high":     65,
        "sma_dist_max": 3.0,
        "sessions":     ["london", "ny"],
        "pip_div":      1,
    },
    "EURUSD": {
        "ticker":       "EURUSD=X",
        "sl_atr":       1.0,
        "tp_atr":       2.5,
        "score_thresh": 60,
        "rsi_low":      35,
        "rsi_high":     65,
        "sma_dist_max": 2.0,
        "sessions":     ["london"],  # EUR/USD is a London pair
        "pip_div":      10000,
    },
    "GBPUSD": {
        "ticker":       "GBPUSD=X",
        "sl_atr":       1.0,
        "tp_atr":       2.5,
        "score_thresh": 60,
        "rsi_low":      35,
        "rsi_high":     65,
        "sma_dist_max": 2.0,
        "sessions":     ["london", "ny"],
        "pip_div":      10000,
    },
}

# ─────────────────────────────────────────
# SESSION WINDOWS (UTC hours)
# ─────────────────────────────────────────
SESSIONS = {
    "asia":   (0,  7),     # Asia range building
    "london": (7,  9),     # London Kill Zone (best entries)
    "london_mid": (9, 12), # London continuation (lower vol)
    "ny":     (13, 16),    # NY Kill Zone (second best)
    "ny_cont":(16, 21),    # NY continuation
}


# ─────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────
def build_indicators(df):
    df = df.copy()
    df['EMA9']   = df['Close'].ewm(span=9).mean()
    df['EMA21']  = df['Close'].ewm(span=21).mean()
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SH50']   = df['High'].rolling(50).max()
    df['SL50']   = df['Low'].rolling(50).min()

    # Asia range (0000-0700 UTC) — detect sweep
    df['Asia_High'] = np.nan
    df['Asia_Low']  = np.nan

    # ATR
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)),
                   abs(df['Low']  - df['Close'].shift(1)))
    )
    df['ATR'] = tr.rolling(14).mean()

    # RSI
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # FVG — 3-candle gap
    df['FVG_Bull'] = df['Low'] > df['High'].shift(2)
    df['FVG_Bear'] = df['High'] < df['Low'].shift(2)

    # OB — impulsive reversal candle
    body = abs(df['Close'] - df['Open'])
    df['OB_Bull'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &  # prev bearish
        (df['Close'] > df['Open']) &                     # curr bullish
        (body > df['ATR'])                               # impulsive
    )
    df['OB_Bear'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'] < df['Open']) &
        (body > df['ATR'])
    )

    # PDH / PDL (Previous Day High/Low)
    daily = df['Close'].resample('D').ohlc()
    df['PDH'] = daily['high'].shift(1).reindex(df.index, method='ffill')
    df['PDL'] = daily['low'].shift(1).reindex(df.index, method='ffill')

    # Liquidity sweep detection (broke PDH/PDL then reversed)
    df['Swept_High'] = (df['High'] > df['PDH']) & (df['Close'] < df['PDH'])
    df['Swept_Low']  = (df['Low']  < df['PDL']) & (df['Close'] > df['PDL'])

    return df.dropna(subset=['ATR', 'RSI', 'EMA9'])


# ─────────────────────────────────────────
# SESSION CHECKER
# ─────────────────────────────────────────
def get_session(hour):
    if  0 <= hour <  7: return "asia"
    if  7 <= hour <  9: return "london"       # Kill Zone
    if  9 <= hour < 12: return "london_mid"
    if 12 <= hour < 13: return "transition"
    if 13 <= hour < 16: return "ny"           # Kill Zone
    if 16 <= hour < 21: return "ny_cont"
    return "dead"

def is_kill_zone(hour):
    return (7 <= hour < 9) or (13 <= hour < 16)


# ─────────────────────────────────────────
# PRICE ACTION SIGNAL SCORER
# ─────────────────────────────────────────
def score_signal(row, bias, cfg):
    """
    Pure price action scoring — no black box.
    Each condition maps to a real trading reason.
    """
    score = 0
    price  = float(row['Close'])
    atr    = float(row['ATR'])
    rsi    = float(row['RSI'])
    hour   = row.name.hour if hasattr(row.name, 'hour') else 0
    session = get_session(hour)

    # ── LAYER 1: BIAS (30 pts) ───────────────
    # EMA9 > EMA21 = momentum bias confirmed
    ema_bull = float(row['EMA9']) > float(row['EMA21'])
    ema_bear = float(row['EMA9']) < float(row['EMA21'])
    if bias == 1  and ema_bull: score += 20
    if bias == -1 and ema_bear: score += 20
    if bias == 1  and price > float(row['SMA50']): score += 10
    if bias == -1 and price < float(row['SMA50']): score += 10

    # Anti-late-trend: penalize if too extended from EMA21
    dist = abs(price - float(row['EMA21'])) / (atr + 1e-9)
    if dist > cfg['sma_dist_max']:
        score -= 15   # Price is too stretched, pullback likely

    # ── LAYER 2: ZONE QUALITY (35 pts) ──────
    sh = float(row['SH50']); sl = float(row['SL50'])
    rng = sh - sl
    zone_pct = (price - sl) / rng if rng > 0 else 0.5

    # Discount = buy zone, Premium = sell zone
    if bias == 1:
        if zone_pct < 0.35: score += 20    # Deep discount (ideal)
        elif zone_pct < 0.50: score += 12  # Upper discount (acceptable)
        else: score -= 10                   # Premium = bad long entry
    else:
        if zone_pct > 0.65: score += 20    # Deep premium (ideal)
        elif zone_pct > 0.50: score += 12  # Lower premium (acceptable)
        else: score -= 10                   # Discount = bad short entry

    # OB / FVG (institutional confirmation)
    if bias == 1 and row.get('OB_Bull', False):  score += 10
    if bias == -1 and row.get('OB_Bear', False): score += 10
    if bias == 1 and row.get('FVG_Bull', False): score += 8
    if bias == -1 and row.get('FVG_Bear', False): score += 8

    # Liquidity sweep (stop hunt before real move)
    if bias == 1  and row.get('Swept_Low',  False): score += 12  # Swept lows → long
    if bias == -1 and row.get('Swept_High', False): score += 12  # Swept highs → short

    # PDH/PDL proximity (key levels)
    pdh = row.get('PDH', 0); pdl = row.get('PDL', 0)
    if pdh and pdl and atr > 0:
        if bias == 1  and abs(price - pdl) < atr * 0.5: score += 8   # Near PDL support
        if bias == -1 and abs(price - pdh) < atr * 0.5: score += 8   # Near PDH resistance

    # ── LAYER 3: SESSION (25 pts) ─────────────
    if session == "london": score += 25     # Best kill zone
    elif session == "ny":   score += 20     # Second best
    elif session == "london_mid": score += 5
    elif session == "transition": score += 5
    elif session == "asia": score -= 20     # Never trade Asia (manipulation zone)
    elif session == "dead": score -= 30     # Dead hours

    # ── LAYER 4: RSI CONFIRMATION ─────────────
    if bias == 1  and rsi < cfg['rsi_low']:  score += 10  # Oversold in uptrend
    if bias == -1 and rsi > cfg['rsi_high']: score += 10  # Overbought in downtrend

    return max(0, min(score, 100))


# ─────────────────────────────────────────
# SIGNAL GENERATOR (Live-style)
# ─────────────────────────────────────────
def generate_signals(pair_name, cfg, mode="backtest"):
    ticker = cfg['ticker']
    print(f"\n[{pair_name}] Loading data ({ticker})...")

    try:
        df_1h = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
        df_1d = yf.download(ticker, period="2y",   interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  ERROR: {e}"); return []

    for df in [df_1h, df_1d]:
        if df is None or df.empty: return []
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Daily bias via EMA cross on daily
    df_1d['EMA9']  = df_1d['Close'].ewm(span=9).mean()
    df_1d['EMA21'] = df_1d['Close'].ewm(span=21).mean()
    df_1d['Bias']  = np.where(
        (df_1d['EMA9'] > df_1d['EMA21']) & (df_1d['Close'] > df_1d['Close'].rolling(50).mean()), 1,
        np.where(
            (df_1d['EMA9'] < df_1d['EMA21']) & (df_1d['Close'] < df_1d['Close'].rolling(50).mean()), -1, 0
        )
    )

    df_1d_bias = df_1d[['Bias']].copy()
    df_1d_bias.index = pd.to_datetime(df_1d_bias.index).normalize().tz_localize(None)
    df_1h.index = pd.to_datetime(df_1h.index).tz_localize(None)

    df_1h = build_indicators(df_1h)
    df_1h['Daily_Bias'] = df_1h.index.normalize().map(df_1d_bias['Bias'].to_dict())
    df_1h['Daily_Bias'] = df_1h['Daily_Bias'].ffill().fillna(0)

    # Backtest loop
    trades = []; active = None
    thresh = cfg['score_thresh']
    sl_m   = cfg['sl_atr']
    tp_m   = cfg['tp_atr']

    for i in range(200, len(df_1h)):
        row   = df_1h.iloc[i]
        atr   = float(row['ATR'])
        price = float(row['Close'])
        bias  = int(row['Daily_Bias'])
        hour  = row.name.hour

        if atr == 0 or bias == 0: continue

        # Only trade configured sessions
        session = get_session(hour)
        pair_sessions = cfg.get('sessions', ['london', 'ny'])
        if session not in pair_sessions: continue

        # Manage trade
        if active:
            hi, lo = float(row['High']), float(row['Low'])
            d = active['dir']
            if d == 1:
                if hi >= active['tp']:
                    active['result'] = 'W'; active['pnl'] = tp_m / sl_m
                    trades.append(active); active = None; continue
                if lo <= active['sl']:
                    active['result'] = 'L'; active['pnl'] = -1.0
                    trades.append(active); active = None; continue
            else:
                if lo <= active['tp']:
                    active['result'] = 'W'; active['pnl'] = tp_m / sl_m
                    trades.append(active); active = None; continue
                if hi >= active['sl']:
                    active['result'] = 'L'; active['pnl'] = -1.0
                    trades.append(active); active = None; continue
            continue

        score = score_signal(row, bias, cfg)
        if score < thresh: continue

        grade = "A" if score >= 75 else ("B" if score >= 60 else "C")

        if bias == 1:
            entry = price; sl = entry - atr*sl_m; tp = entry + atr*tp_m
        else:
            entry = price; sl = entry + atr*sl_m; tp = entry - atr*tp_m

        active = {
            'time': row.name, 'pair': pair_name,
            'dir': bias, 'entry': entry, 'sl': sl, 'tp': tp,
            'score': score, 'grade': grade, 'session': session,
            'pnl': 0, 'result': None
        }

    return trades


# ─────────────────────────────────────────
# RUN ALL PAIRS + REPORT
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  FINAL SIGNAL ENGINE v4 — Price Action + Session")
    print("=" * 65)

    all_trades = []
    for pair, cfg in PAIR_CONFIGS.items():
        trades = generate_signals(pair, cfg)
        all_trades.extend(trades)

        if trades:
            wins   = [t for t in trades if t['result'] == 'W']
            total  = len(trades)
            wr     = len(wins) / total * 100
            net_r  = sum(t['pnl'] for t in trades)
            pf_n   = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            pf_d   = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            pf     = pf_n / max(pf_d, 0.001)
            rr     = cfg['tp_atr'] / cfg['sl_atr']
            wk     = total / (730/7)

            # Grade breakdown
            for g in ['A', 'B', 'C']:
                g_trades = [t for t in trades if t['grade'] == g]
                g_wins   = [t for t in g_trades if t['result'] == 'W']
                if g_trades:
                    g_wr = len(g_wins) / len(g_trades) * 100
                    print(f"  Grade {g}: {len(g_trades):3d} trades | WR {g_wr:5.1f}%")

            print(f"\n  [{pair}] Trades: {total} | WR: {wr:.1f}% | "
                  f"RRR: 1:{rr:.1f} | Net R: {net_r:+.1f} | "
                  f"PF: {pf:.2f} | {wk:.1f}/wk")

    # Combined
    if all_trades:
        total_t = len(all_trades)
        total_w = len([t for t in all_trades if t['result'] == 'W'])
        total_r = sum(t['pnl'] for t in all_trades)
        print("\n" + "=" * 65)
        print(f"  COMBINED PORTFOLIO")
        print(f"  Total Trades : {total_t}")
        print(f"  Win Rate     : {total_w/total_t*100:.1f}%")
        print(f"  Net R (2yr)  : {total_r:+.1f}R ({total_r:+.1f}% @ 1% risk)")
        print(f"  Signals/wk   : {total_t/(730/7):.1f}")
        print("=" * 65)
