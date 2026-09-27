import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from datetime import datetime, timezone
from notifier import send_discord_alert
from local_db import save_signal

# ─────────────────────────────────────────
# CONFIG: SWING & POSITIONAL ENGINE
# ─────────────────────────────────────────
PAIRS = {
    "US30":   {"ticker": "YM=F",      "sl_atr": 2.0, "tp_atr": 6.0},
    "NAS100": {"ticker": "NQ=F",      "sl_atr": 2.0, "tp_atr": 6.0},
    "US500":  {"ticker": "ES=F",      "sl_atr": 2.0, "tp_atr": 6.0},
    "Gold":   {"ticker": "GC=F",      "sl_atr": 2.5, "tp_atr": 8.0},
    "Silver": {"ticker": "SI=F",      "sl_atr": 2.5, "tp_atr": 8.0},
    "Oil":    {"ticker": "CL=F",      "sl_atr": 2.5, "tp_atr": 8.0},
    "BTC":    {"ticker": "BTC-USD",   "sl_atr": 2.5, "tp_atr": 10.0},
    "ETH":    {"ticker": "ETH-USD",   "sl_atr": 2.5, "tp_atr": 10.0},
    "EURUSD": {"ticker": "EURUSD=X",  "sl_atr": 1.5, "tp_atr": 5.0},
    "GBPUSD": {"ticker": "GBPUSD=X",  "sl_atr": 1.5, "tp_atr": 5.0},
    "USDJPY": {"ticker": "JPY=X",     "sl_atr": 1.5, "tp_atr": 5.0},
}

MODELS_DIR = "models"
models = {}

def load_swing_models():
    global models
    for pair in PAIRS:
        path = f"{MODELS_DIR}/swing_{pair}_model.json"
        if os.path.exists(path):
            m = xgb.XGBClassifier()
            m.load_model(path)
            models[pair] = m
            
# ─────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────
def get_swing_data(pair, cfg):
    """Fetch Weekly (1wk) and Daily (1d) data for Swing/Positional Trades."""
    try:
        t = yf.Ticker(cfg['ticker'])
        df_1wk = t.history(period="2y", interval="1wk")
        df_1d  = t.history(period="1y", interval="1d")
        
        if df_1wk.empty or df_1d.empty: return None, None
        
        # Weekly Bias
        df_1wk['EMA9_W'] = df_1wk['Close'].ewm(span=9).mean()
        df_1wk['EMA21_W'] = df_1wk['Close'].ewm(span=21).mean()
        df_1wk['Bias_W'] = np.where(df_1wk['EMA9_W'] > df_1wk['EMA21_W'], 1, -1)
        
        # Daily Features
        df_1d['EMA9'] = df_1d['Close'].ewm(span=9).mean()
        df_1d['EMA21'] = df_1d['Close'].ewm(span=21).mean()
        df_1d['SMA50'] = df_1d['Close'].rolling(50).mean()
        
        tr = np.maximum(df_1d['High'] - df_1d['Low'], 
             np.maximum(abs(df_1d['High'] - df_1d['Close'].shift(1)), 
                        abs(df_1d['Low'] - df_1d['Close'].shift(1))))
        df_1d['ATR'] = tr.rolling(14).mean()
        
        df_1d['RSI'] = 100 - (100 / (1 + df_1d['Close'].diff().clip(lower=0).rolling(14).mean() / (-df_1d['Close'].diff().clip(upper=0)).rolling(14).mean().replace(0, np.nan)))
        
        # Map Weekly bias to Daily
        df_1wk_bias = df_1wk[['Bias_W']].copy()
        df_1wk_bias.index = pd.to_datetime(df_1wk_bias.index).normalize().tz_localize(None)
        
        df_1d.index = pd.to_datetime(df_1d.index).normalize().tz_localize(None)
        
        # Merge bias
        latest_weekly_bias = int(df_1wk['Bias_W'].iloc[-1])
        df_1d['Weekly_Bias'] = latest_weekly_bias
        
        df_1d = df_1d.dropna()
        if df_1d.empty: return None, None
        
        return df_1d.iloc[-1].copy(), df_1d

    except Exception as e:
        print(f"Error fetching {pair}: {e}")
        return None, None

def run_swing_engine():
    print("=" * 55)
    print("  SWING ENGINE — v4.1 PRO MAX / v5 PRO MAX")
    print("  Targeting: Swing & Positional Trades (Daily Timeframe)")
    print("=" * 55)
    
    load_swing_models()
    now_utc = datetime.now(timezone.utc)
    
    for pair, cfg in PAIRS.items():
        row, df = get_swing_data(pair, cfg)
        if row is None: continue
        
        bias = int(row['Weekly_Bias'])
        price = float(row['Close'])
        atr = float(row['ATR'])
        
        # V4.1 PRO MAX Rules Score
        score = 50 
        if bias == 1 and price > row['SMA50']: score += 25
        if bias == -1 and price < row['SMA50']: score += 25
        if row['RSI'] < 40 and bias == 1: score += 15
        if row['RSI'] > 60 and bias == -1: score += 15
        
        if score >= 75:
            # V5 PRO MAX AI Probability
            win_prob = 0.5
            if pair in models:
                feat_cols = ['Weekly_Bias', 'RSI', 'EMA9', 'EMA21', 'SMA50']
                feat_vec = [[row.get(f, 0) for f in feat_cols]]
                win_prob = float(models[pair].predict_proba(feat_vec)[0][1])
                
            if win_prob < 0.52:
                print(f"  [{pair}] Rules passed but ML rejected (P={win_prob:.2f})")
                continue

            direction = "LONG" if bias == 1 else "SHORT"
            sl_atr = cfg['sl_atr']
            tp_atr = cfg['tp_atr']
            
            trade_type = "Positional (1-4 Weeks)" if tp_atr >= 8.0 else "Swing Trade (2-5 Days)"
            
            sl = price - atr * sl_atr if bias == 1 else price + atr * sl_atr
            tp = price + atr * tp_atr if bias == 1 else price - atr * tp_atr
            
            signal = {
                "pair": pair,
                "direction": direction,
                "entry": round(price, 5),
                "sl": round(sl, 5),
                "tp": round(tp, 5),
                "grade": "A" if score >= 85 else "B",
                "score": score,
                "win_prob": round(win_prob, 3),
                "session": "Daily Close",
                "rr": round(tp_atr / sl_atr, 2),
                "time_utc": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
                "engine": f"v4.1/v5 PRO MAX | {trade_type}"
            }
            
            print(f"  [SWING SIGNAL] {pair} {direction} | Win Prob: {win_prob*100:.1f}%")
            send_discord_alert(signal)
            save_signal(signal)
            time.sleep(1)

if __name__ == "__main__":
    run_swing_engine()
    print("  Swing Engine scan complete.")
