import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# ─────────────────────────────────────────
# CONFIG: GURU TRAINING (5m Timeframe)
# ─────────────────────────────────────────
PAIRS = {
    "Gold":   {"ticker": "GC=F", "sl_atr": 1.0, "tp_atr": 3.0},
    "US30":   {"ticker": "YM=F", "sl_atr": 1.0, "tp_atr": 3.0},
    "NAS100": {"ticker": "NQ=F", "sl_atr": 1.0, "tp_atr": 3.0},
    "BTC":    {"ticker": "BTC-USD", "sl_atr": 1.5, "tp_atr": 4.5}
}

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def build_guru_features(df):
    df = df.copy()
    
    # ATR
    tr = np.maximum(df['High'] - df['Low'],
         np.maximum(abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low']  - df['Close'].shift(1))))
    df['ATR'] = tr.rolling(14).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    
    # Imbalance / Momentum
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Body_Norm'] = df['Body'] / df['ATR']
    
    # Simple Trend
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    df['Trend'] = np.where(df['EMA9'] > df['EMA21'], 1, -1)
    
    df = df.dropna()
    return df

# ─────────────────────────────────────────
# LABELING
# ─────────────────────────────────────────
def label_guru_trades(df, sl_atr, tp_atr):
    labels = []
    prices = df['Close'].values
    highs  = df['High'].values
    lows   = df['Low'].values
    atrs   = df['ATR'].values
    trends = df['Trend'].values

    # Look forward up to 30 candles (2.5 hours)
    for i in range(len(df) - 30):
        entry = prices[i]
        atr   = atrs[i]
        trend = trends[i]
        
        # If trend is up, assume we are looking for LONG
        if trend == 1:
            sl = entry - (atr * sl_atr)
            tp = entry + (atr * tp_atr)
            win = 0
            for j in range(i+1, i+30):
                if lows[j] <= sl:
                    break
                if highs[j] >= tp:
                    win = 1
                    break
            labels.append(win)
        
        # If trend is down, assume we are looking for SHORT
        else:
            sl = entry + (atr * sl_atr)
            tp = entry - (atr * tp_atr)
            win = 0
            for j in range(i+1, i+30):
                if highs[j] >= sl:
                    break
                if lows[j] <= tp:
                    win = 1
                    break
            labels.append(win)
            
    labels.extend([np.nan] * 30)
    df['Target'] = labels
    return df.dropna(subset=['Target'])

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
def train_guru_models():
    print("="*50)
    print(" TRAINING GURU AI MODELS (Adaptive Layer)")
    print("="*50)
    
    feature_cols = ['RSI', 'Body_Norm', 'Trend']
    
    for pair_name, cfg in PAIRS.items():
        print(f"Training Guru AI for {pair_name}...")
        try:
            t = yf.Ticker(cfg['ticker'])
            # 60 days of 5m data is the max allowed by yfinance
            df = t.history(period="60d", interval="5m")
            
            if len(df) < 500:
                print(f"  Not enough data for {pair_name}")
                continue
                
            df_feat = build_guru_features(df)
            df_labeled = label_guru_trades(df_feat, cfg['sl_atr'], cfg['tp_atr'])
            
            if df_labeled.empty:
                continue

            X = df_labeled[feature_cols]
            y = df_labeled['Target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                eval_metric='logloss',
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            
            model_path = f"{MODELS_DIR}/guru_{pair_name}_model.json"
            model.save_model(model_path)
            print(f"  {pair_name} | Accuracy: {acc*100:.1f}% | Saved: {model_path}")
            
        except Exception as e:
            print(f"  Failed on {pair_name}: {e}")

if __name__ == "__main__":
    train_guru_models()
    print("\nTraining Complete! Guru Bots are now self-learning.")
