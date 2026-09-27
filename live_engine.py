"""
LIVE ENGINE — v4.1 + v5 Combined Signal System
===============================================
Runs on your PC during London/NY sessions.
Every 5 minutes:
  1. Gets latest candle data per pair
  2. v4.1 rules engine scores the setup
  3. v5 XGBoost predicts win probability
  4. If BOTH agree → fires signal (email + Supabase)

Run: python live_engine.py
"""

import os
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from datetime import datetime, timezone
from notifier import send_discord_alert
from local_db import save_signal

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
PAIRS = {
    "US30":   {"ticker": "YM=F",      "sl_atr": 1.2, "tp_atr": 3.0},
    "NAS100": {"ticker": "NQ=F",      "sl_atr": 1.2, "tp_atr": 3.0},
    "Gold":   {"ticker": "GC=F",      "sl_atr": 1.5, "tp_atr": 3.0},
    "EURUSD": {"ticker": "EURUSD=X",  "sl_atr": 1.0, "tp_atr": 2.5},
    "GBPUSD": {"ticker": "GBPUSD=X",  "sl_atr": 1.0, "tp_atr": 2.5},
}

MODELS_DIR      = "models"
V41_SCORE_MIN   = 55     # v4.1 minimum score to pass
V5_PROB_MIN     = 0.52   # v5 minimum win probability to pass
SCAN_INTERVAL   = 300    # seconds between scans (5 min)
GRADE_A_CUTOFF  = 75     # Suppress Grade A (paradox fix from v4)
KILL_ZONES_UTC  = [(7, 9), (13, 16)]  # London & NY

FEATURE_COLS = [
    'Daily_Bias', 'Session', 'Zone_Pct', 'EMA_Dist',
    'Price_SMA50', 'EMA_Cross', 'RSI_norm',
    'FVG_Bull', 'FVG_Bear', 'OB_Bull', 'OB_Bear',
    'Swept_High', 'Swept_Low'
]

# ─────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────
models = {}

def load_all_models():
    global models
    for pair in PAIRS:
        path = f"{MODELS_DIR}/{pair}_model.json"
        if os.path.exists(path):
            m = xgb.XGBClassifier()
            m.load_model(path)
            models[pair] = m

            meta_path = f"{MODELS_DIR}/{pair}_meta.json"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                print(f"  [{pair}] Model loaded — trained {meta['trained_on']} | "
                      f"test acc: {meta['test_acc']:.1%} | {meta['status']}")
        else:
            print(f"  [{pair}] No model found — run train_v5.py first")


# ─────────────────────────────────────────
# SESSION HELPERS
# ─────────────────────────────────────────
def session_code(hour):
    if  7 <= hour <  9: return 3
    if 13 <= hour < 16: return 2
    if  9 <= hour < 12: return 1
    if  0 <= hour <  7: return -1
    return 0

def is_kill_zone(hour):
    return any(lo <= hour < hi for lo, hi in KILL_ZONES_UTC)

def get_session_name(hour):
    if  7 <= hour <  9: return "London Kill Zone"
    if 13 <= hour < 16: return "NY Kill Zone"
    if  9 <= hour < 12: return "London Mid"
    if  0 <= hour <  7: return "Asia (avoid)"
    return "Off-hours"


# ─────────────────────────────────────────
# DATA + FEATURE EXTRACTION
# ─────────────────────────────────────────
def get_features(pair, cfg):
    ticker = cfg['ticker']
    try:
        df_1h = yf.download(ticker, period="60d", interval="1h",
                            progress=False, auto_adjust=True)
        df_1d = yf.download(ticker, period="180d", interval="1d",
                            progress=False, auto_adjust=True)
    except Exception as e:
        return None, None

    for df in [df_1h, df_1d]:
        if df is None or df.empty: return None, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    df_1h.index = pd.to_datetime(df_1h.index).tz_localize(None)
    df_1d.index = pd.to_datetime(df_1d.index).tz_localize(None)

    # Indicators
    df_1h['EMA9']  = df_1h['Close'].ewm(span=9).mean()
    df_1h['EMA21'] = df_1h['Close'].ewm(span=21).mean()
    df_1h['SMA50'] = df_1h['Close'].rolling(50).mean()
    df_1h['SH50']  = df_1h['High'].rolling(50).max()
    df_1h['SL50']  = df_1h['Low'].rolling(50).min()

    tr = np.maximum(df_1h['High'] - df_1h['Low'],
         np.maximum(abs(df_1h['High'] - df_1h['Close'].shift(1)),
                    abs(df_1h['Low']  - df_1h['Close'].shift(1))))
    df_1h['ATR'] = tr.rolling(14).mean()

    delta = df_1h['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df_1h['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    body = abs(df_1h['Close'] - df_1h['Open'])
    df_1h['FVG_Bull'] = (df_1h['Low'] > df_1h['High'].shift(2)).astype(int)
    df_1h['FVG_Bear'] = (df_1h['High'] < df_1h['Low'].shift(2)).astype(int)
    df_1h['OB_Bull']  = ((df_1h['Close'].shift(1) < df_1h['Open'].shift(1)) &
                          (df_1h['Close'] > df_1h['Open']) & (body > df_1h['ATR'])).astype(int)
    df_1h['OB_Bear']  = ((df_1h['Close'].shift(1) > df_1h['Open'].shift(1)) &
                          (df_1h['Close'] < df_1h['Open']) & (body > df_1h['ATR'])).astype(int)

    daily_high = df_1h['High'].resample('D').max()
    daily_low  = df_1h['Low'].resample('D').min()
    df_1h['PDH'] = daily_high.shift(1).reindex(df_1h.index, method='ffill')
    df_1h['PDL'] = daily_low.shift(1).reindex(df_1h.index, method='ffill')
    df_1h['Swept_High'] = ((df_1h['High'] > df_1h['PDH']) & (df_1h['Close'] < df_1h['PDH'])).astype(int)
    df_1h['Swept_Low']  = ((df_1h['Low'] < df_1h['PDL'])  & (df_1h['Close'] > df_1h['PDL'])).astype(int)

    # Daily bias
    df_1d['EMA9d']  = df_1d['Close'].ewm(span=9).mean()
    df_1d['EMA21d'] = df_1d['Close'].ewm(span=21).mean()
    df_1d['SMA50d'] = df_1d['Close'].rolling(50).mean()
    df_1d['Bias']   = np.where(
        (df_1d['EMA9d'] > df_1d['EMA21d']) & (df_1d['Close'] > df_1d['SMA50d']), 1,
        np.where((df_1d['EMA9d'] < df_1d['EMA21d']) & (df_1d['Close'] < df_1d['SMA50d']), -1, 0)
    )
    df_1d_bias = df_1d[['Bias']].copy()
    df_1d_bias.index = pd.to_datetime(df_1d_bias.index).normalize().tz_localize(None)
    df_1h['Daily_Bias'] = df_1h.index.normalize().map(df_1d_bias['Bias'].to_dict())
    df_1h['Daily_Bias'] = df_1h['Daily_Bias'].ffill().fillna(0)

    df_1h = df_1h.dropna(subset=['ATR', 'RSI', 'EMA9'])
    if df_1h.empty: return None, None

    latest = df_1h.iloc[-1].copy()
    price  = float(latest['Close'])
    atr    = float(latest['ATR'])
    sh, sl = float(latest['SH50']), float(latest['SL50'])
    rng    = sh - sl

    latest['Zone_Pct']    = (price - sl) / rng if rng > 0 else 0.5
    latest['EMA_Dist']    = abs(price - float(latest['EMA21'])) / (atr + 1e-9)
    latest['Price_SMA50'] = (price - float(latest['SMA50'])) / (atr + 1e-9)
    latest['EMA_Cross']   = int(float(latest['EMA9']) > float(latest['EMA21']))
    latest['RSI_norm']    = float(latest['RSI']) / 100
    latest['Session']     = session_code(latest.name.hour)

    return latest, df_1h


# ─────────────────────────────────────────
# V4.1 RULES SCORER
# ─────────────────────────────────────────
def v41_score(row, bias):
    score = 0
    price  = float(row['Close'])
    atr    = float(row['ATR'])
    rsi    = float(row['RSI'])
    hour   = row.name.hour

    # Bias via EMA
    ema_bull = float(row['EMA9']) > float(row['EMA21'])
    if bias == 1  and ema_bull:  score += 20
    if bias == -1 and not ema_bull: score += 20
    if bias == 1  and price > float(row['SMA50']): score += 10
    if bias == -1 and price < float(row['SMA50']): score += 10

    # Anti-late-trend
    dist = abs(price - float(row['EMA21'])) / (atr + 1e-9)
    if dist > 2.5: score -= 15

    # Premium/Discount zone
    sh = float(row['SH50']); sl = float(row['SL50'])
    rng = sh - sl
    if rng > 0:
        zone_pct = (price - sl) / rng
        if bias == 1:
            if zone_pct < 0.35:   score += 20
            elif zone_pct < 0.50: score += 12
            else:                  score -= 10
        else:
            if zone_pct > 0.65:   score += 20
            elif zone_pct > 0.50: score += 12
            else:                  score -= 10

    # OB / FVG / Sweep
    if bias == 1  and row.get('OB_Bull', 0):    score += 10
    if bias == -1 and row.get('OB_Bear', 0):    score += 10
    if bias == 1  and row.get('FVG_Bull', 0):   score += 8
    if bias == -1 and row.get('FVG_Bear', 0):   score += 8
    if bias == 1  and row.get('Swept_Low', 0):  score += 12
    if bias == -1 and row.get('Swept_High', 0): score += 12

    # Session
    sess = session_code(hour)
    if   sess == 3:  score += 25   # London kill zone
    elif sess == 2:  score += 20   # NY kill zone
    elif sess == 1:  score += 5
    elif sess == -1: score -= 20   # Asia — penalize hard
    else:            score -= 30

    # RSI
    if bias == 1  and rsi < 40: score += 10
    if bias == -1 and rsi > 60: score += 10

    return max(0, min(score, 100))


# ─────────────────────────────────────────
# MAIN SCAN LOOP
# ─────────────────────────────────────────
fired_today = set()   # Prevent duplicate signals same candle

def scan_all_pairs():
    now_utc = datetime.now(timezone.utc)
    hour    = now_utc.hour

    # Only scan during kill zones
    if not is_kill_zone(hour):
        session = get_session_name(hour)
        print(f"  [{now_utc.strftime('%H:%M')} UTC] Outside kill zone ({session}) — waiting...")
        return

    print(f"\n  [{now_utc.strftime('%H:%M')} UTC] SCANNING — {get_session_name(hour)}")

    for pair, cfg in PAIRS.items():
        try:
            row, df = get_features(pair, cfg)
            if row is None: continue

            bias = int(row['Daily_Bias'])
            if bias == 0: continue

            # V4.1 score
            score = v41_score(row, bias)
            if score < V41_SCORE_MIN:
                continue

            # Skip Grade A (paradox fix)
            if score >= GRADE_A_CUTOFF:
                continue

            grade = "B" if score >= 60 else "C"

            # V5 probability
            win_prob = 0.5
            if pair in models:
                feat_vec = [[row.get(f, 0) for f in FEATURE_COLS]]
                win_prob = float(models[pair].predict_proba(feat_vec)[0][1])

            if win_prob < V5_PROB_MIN:
                print(f"    {pair}: v4.1 passed ({score}) but v5 rejected (P={win_prob:.2f})")
                continue

            # Both engines agree — FIRE
            price  = float(row['Close'])
            atr    = float(row['ATR'])
            direction = "LONG" if bias == 1 else "SHORT"
            sl = price - atr * cfg['sl_atr'] if bias == 1 else price + atr * cfg['sl_atr']
            tp = price + atr * cfg['tp_atr'] if bias == 1 else price - atr * cfg['tp_atr']

            signal_key = f"{pair}_{now_utc.strftime('%Y%m%d%H')}"
            if signal_key in fired_today:
                continue
            fired_today.add(signal_key)

            signal = {
                "pair":      pair,
                "direction": direction,
                "entry":     round(price, 5),
                "sl":        round(sl, 5),
                "tp":        round(tp, 5),
                "grade":     grade,
                "score":     score,
                "win_prob":  round(win_prob, 3),
                "session":   get_session_name(now_utc.hour),
                "rr":        round(cfg['tp_atr'] / cfg['sl_atr'], 2),
                "time_utc":  now_utc.strftime("%Y-%m-%d %H:%M UTC")
            }

            print(f"\n  *** SIGNAL FIRED ***")
            print(f"  Pair:      {signal['pair']}")
            print(f"  Direction: {signal['direction']}")
            print(f"  Grade:     {signal['grade']} (score {signal['score']})")
            print(f"  Win Prob:  {signal['win_prob']*100:.1f}% (v5 ML)")
            print(f"  Entry:     {signal['entry']}")
            print(f"  SL:        {signal['sl']}")
            print(f"  TP:        {signal['tp']} (1:{signal['rr']} RRR)")
            print(f"  Session:   {signal['session']}")

            send_discord_alert(signal)
            save_signal(signal)

        except Exception as e:
            print(f"    {pair}: Error — {e}")


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  LIVE ENGINE — v4.1 + v5 Combined")
    print("  Scanning London (0700-0930) + NY (1330-1600) UTC")
    print("=" * 55)
    print("\nLoading ML models...")
    load_all_models()
    print("\nEngine started. Scanning every 5 minutes...\n")

    while True:
        try:
            scan_all_pairs()
        except KeyboardInterrupt:
            print("\nEngine stopped by user.")
            break
        except Exception as e:
            print(f"  Scan error: {e}")
        time.sleep(SCAN_INTERVAL)
