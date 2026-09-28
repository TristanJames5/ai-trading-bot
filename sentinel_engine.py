"""
SENTINEL HYBRID ENGINE
======================
Runs on your PC during London/NY sessions.
Every 5 minutes:
  1. Gets latest candle data per pair
  2. XGBoost (Analyst) predicts win probabilities
  3. PPO (Fund Manager) takes Analyst's probabilities + open positions
  4. PPO decides: Risk Sizing (1%, 5%, 10%) or Close Early

Run: python sentinel_engine.py
"""

import os
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from datetime import datetime, timezone
from stable_baselines3 import PPO
from dotenv import load_dotenv

# Load environment variables (like DISCORD_WEBHOOK_URL)
load_dotenv()

# Internal modules
from notifier import send_discord_alert, send_discord_result
from local_db import save_signal, get_active_signals, log_trade_result

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
PAIRS = {
    # Forex
    "EURUSD": {"ticker": "EURUSD=X",  "sl_atr": 1.0, "tp_atr": 2.5},
    "GBPUSD": {"ticker": "GBPUSD=X",  "sl_atr": 1.0, "tp_atr": 2.5},
    
    # Indices
    "US30":   {"ticker": "YM=F",      "sl_atr": 1.2, "tp_atr": 3.0},
    "NAS100": {"ticker": "NQ=F",      "sl_atr": 1.2, "tp_atr": 3.0},
    "US500":  {"ticker": "ES=F",      "sl_atr": 1.2, "tp_atr": 3.0},

    # Crypto & Commodities
    "Gold":   {"ticker": "GC=F",      "sl_atr": 1.5, "tp_atr": 3.0},
    "BTC":    {"ticker": "BTC-USD",   "sl_atr": 1.5, "tp_atr": 3.0},
}

MODELS_DIR      = "models"
SCAN_INTERVAL   = 300    # seconds between scans (5 min)
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
xgb_models = {}
ppo_model = None

def load_all_models():
    global xgb_models, ppo_model
    
    # Load XGBoost (The Analyst)
    for pair in PAIRS:
        path = f"{MODELS_DIR}/{pair}_model.json"
        if os.path.exists(path):
            m = xgb.XGBClassifier()
            m.load_model(path)
            xgb_models[pair] = m
            print(f"  [{pair}] XGBoost Analyst loaded")
        else:
            print(f"  [{pair}] No XGBoost Analyst found — using 50/50 fallback")
            
    # Load PPO (The Fund Manager)
    if os.path.exists("sentinel_model.zip"):
        print("\n  🛡️ Loading PPO Fund Manager (Sentinel Brain)...")
        # We don't need a real env to just run inference
        ppo_model = PPO.load("sentinel_model.zip")
        print("  ✅ Sentinel Brain Online")
    else:
        print("\n  ❌ ERROR: sentinel_model.zip not found!")
        exit(1)


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
# MAIN SCAN LOOP
# ─────────────────────────────────────────
fired_today = set()

def check_active_trades():
    """Check open signals for SL/TP hits (Standard logic)"""
    active_signals = get_active_signals()
    if not active_signals: return

    for sig in active_signals:
        pair = sig['pair']
        if pair not in PAIRS: continue
        ticker = PAIRS[pair]['ticker']
        try:
            df = yf.Ticker(ticker).history(period="1d", interval="5m")
            if df.empty: continue
            current_price = float(df['Close'].iloc[-1])
            
            if sig['direction'] == "LONG":
                if current_price >= sig['tp']:
                    log_trade_result(sig['id'], "WIN", sig['rr'], "Hit TP")
                    send_discord_result(sig, "WIN", sig['rr'])
                elif current_price <= sig['sl']:
                    log_trade_result(sig['id'], "LOSS", -1.0, "Hit SL")
                    send_discord_result(sig, "LOSS", -1.0)
            elif sig['direction'] == "SHORT":
                if current_price <= sig['tp']:
                    log_trade_result(sig['id'], "WIN", sig['rr'], "Hit TP")
                    send_discord_result(sig, "WIN", sig['rr'])
                elif current_price >= sig['sl']:
                    log_trade_result(sig['id'], "LOSS", -1.0, "Hit SL")
                    send_discord_result(sig, "LOSS", -1.0)
        except Exception as e:
            pass

def scan_all_pairs():
    now_utc = datetime.now(timezone.utc)
    hour    = now_utc.hour

    if not is_kill_zone(hour):
        session = get_session_name(hour)
        print(f"  [{now_utc.strftime('%H:%M')} UTC] Outside kill zone ({session}) — waiting...")
        return

    print(f"\n  [{now_utc.strftime('%H:%M')} UTC] SENTINEL SCANNING — {get_session_name(hour)}")
    active_signals = get_active_signals()

    for pair, cfg in PAIRS.items():
        try:
            row, df = get_features(pair, cfg)
            if row is None: continue

            # 1. Get XGBoost Probabilities (The Analyst)
            long_prob, short_prob = 0.5, 0.5
            if pair in xgb_models:
                feat_vec = [[row.get(f, 0) for f in FEATURE_COLS]]
                probs = xgb_models[pair].predict_proba(feat_vec)[0]
                short_prob, long_prob = probs[0], probs[1]
                
            atr = float(row['ATR'])
            current_price = float(row['Close'])
            
            # 2. Find Open Position for this pair
            position_flag = 0  # 0=flat, 1=long, -1=short
            unrealized_pnl = 0.0
            open_trade = None
            
            for sig in active_signals:
                if sig['pair'] == pair:
                    open_trade = sig
                    position_flag = 1 if sig['direction'] == "LONG" else -1
                    entry = float(sig['entry'])
                    if position_flag == 1:
                        unrealized_pnl = (current_price - entry) / entry * 100.0
                    else:
                        unrealized_pnl = (entry - current_price) / entry * 100.0
                    break

            # 3. Create Sentinel Observation
            # [Long_Prob, Short_Prob, ATR, unrealized_pnl_pct, position]
            obs = np.array([long_prob, short_prob, atr, unrealized_pnl, position_flag], dtype=np.float32)
            
            # 4. PPO (Fund Manager) Decides Action
            action, _ = ppo_model.predict(obs, deterministic=True)
            action = int(action)
            
            print(f"    {pair}: L:{long_prob:.2f} S:{short_prob:.2f} | Pos:{position_flag} PnL:{unrealized_pnl:.2f}% -> Sentinel Action: {action}")
            
            # ACTION MAPPING:
            # 0: Hold Flat
            # 1: Buy Light (1%), 2: Buy Normal (5%), 3: Buy MAX (10%)
            # 4: Sell Light (1%), 5: Sell Normal (5%), 6: Sell MAX (10%)
            # 7: Close Position
            
            if action == 7 and position_flag != 0 and open_trade:
                print(f"    🚨 SENTINEL COMMANDS EARLY CLOSE on {pair}!")
                result = "WIN" if unrealized_pnl > 0 else "LOSS"
                # Roughly estimate RR closed early
                pseudo_rr = round(unrealized_pnl / 1.0, 2) if unrealized_pnl > 0 else -1.0 
                log_trade_result(open_trade['id'], result, pseudo_rr, "Sentinel Early Close")
                send_discord_result(open_trade, result, pseudo_rr)
                continue
                
            if position_flag == 0 and action in [1, 2, 3, 4, 5, 6]:
                # Sentinel wants to enter a new trade
                direction = "LONG" if action in [1, 2, 3] else "SHORT"
                risk_level = {1: "Light (1%)", 2: "Normal (5%)", 3: "MAX (10%)",
                              4: "Light (1%)", 5: "Normal (5%)", 6: "MAX (10%)"}[action]
                              
                sl_std = current_price - atr * cfg['sl_atr'] if direction == "LONG" else current_price + atr * cfg['sl_atr']
                tp_std = current_price + atr * cfg['tp_atr'] if direction == "LONG" else current_price - atr * cfg['tp_atr']

                signal_key = f"{pair}_{now_utc.strftime('%Y%m%d%H')}"
                if signal_key in fired_today:
                    continue
                fired_today.add(signal_key)

                signal_data = {
                    "pair":      pair,
                    "direction": direction,
                    "entry":     round(current_price, 5),
                    "sl":        round(sl_std, 5),
                    "tp":        round(tp_std, 5),
                    "grade":     "Sentinel",
                    "score":     int(max(long_prob, short_prob) * 100),
                    "win_prob":  round(max(long_prob, short_prob), 3),
                    "session":   get_session_name(now_utc.hour),
                    "rr":        round(cfg['tp_atr'] / cfg['sl_atr'], 2),
                    "time_utc":  now_utc.strftime("%Y-%m-%d %H:%M UTC"),
                    "engine":    f"Sentinel AI [{risk_level}]"
                }

                print(f"\n  *** 🛡️ SENTINEL FIRED SIGNAL ***")
                print(f"  Pair:      {signal_data['pair']} ({signal_data['direction']})")
                print(f"  Risk:      {risk_level}")
                print(f"  Entry:     {signal_data['entry']}")
                print(f"  SL:        {signal_data['sl']} | TP: {signal_data['tp']} (1:{signal_data['rr']})")

                send_discord_alert(signal_data)
                save_signal(signal_data)
                
        except Exception as e:
            print(f"    {pair}: Error — {e}")


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🛡️ APEX SENTINEL — LIVE HYBRID ENGINE")
    print("  Scanning London (0700-0930) + NY (1330-1600) UTC")
    print("=" * 55)
    load_all_models()
    print("\nSentinel Engine started. Scanning every 5 minutes...\n")

    while True:
        try:
            check_active_trades()
            scan_all_pairs()
        except KeyboardInterrupt:
            print("\nSentinel Engine stopped by user.")
            break
        except Exception as e:
            print(f"  Scan error: {e}")
        time.sleep(SCAN_INTERVAL)
