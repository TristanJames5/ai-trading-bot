import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# ─────────────────────────────────────────
# CONFIG: SWING & POSITIONAL TRAINING
# ─────────────────────────────────────────
PAIRS = {
    "US30":   "YM=F", "NAS100": "NQ=F", "US500":  "ES=F",
    "Gold":   "GC=F", "Silver": "SI=F", "Oil":    "CL=F",
    "BTC":    "BTC-USD", "ETH": "ETH-USD",
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X"
}

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# FEATURE ENGINEERING (DAILY)
# ─────────────────────────────────────────
def build_swing_features(df, df_1wk):
    df = df.copy()

    # Trend
    df['EMA9']   = df['Close'].ewm(span=9).mean()
    df['EMA21']  = df['Close'].ewm(span=21).mean()
    df['SMA50']  = df['Close'].rolling(50).mean()
    
    # Weekly Bias
    df_1wk['EMA9_W'] = df_1wk['Close'].ewm(span=9).mean()
    df_1wk['EMA21_W'] = df_1wk['Close'].ewm(span=21).mean()
    df_1wk['Bias_W'] = np.where(df_1wk['EMA9_W'] > df_1wk['EMA21_W'], 1, -1)
    
    df_1wk_bias = df_1wk[['Bias_W']].copy()
    df_1wk_bias.index = pd.to_datetime(df_1wk_bias.index).normalize().tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
    
    # Map Weekly Bias to Daily
    mapped_bias = pd.Series(df.index.map(df_1wk_bias['Bias_W'].to_dict()))
    df['Weekly_Bias'] = mapped_bias.ffill().fillna(0).values

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

    df = df.dropna()
    return df

# ─────────────────────────────────────────
# LABELING (1:3 RRR SWING)
# ─────────────────────────────────────────
def label_swing_trades(df):
    labels = []
    prices = df['Close'].values
    highs  = df['High'].values
    lows   = df['Low'].values
    atrs   = df['ATR'].values
    biases = df['Weekly_Bias'].values

    for i in range(len(df) - 20):
        entry = prices[i]
        atr   = atrs[i]
        bias  = biases[i]

        if bias == 1:
            sl = entry - (atr * 2.0)
            tp = entry + (atr * 6.0)
            win = 0
            for j in range(i+1, min(i+20, len(df))):
                if lows[j] <= sl:
                    break
                if highs[j] >= tp:
                    win = 1
                    break
            labels.append(win)
        
        elif bias == -1:
            sl = entry + (atr * 2.0)
            tp = entry - (atr * 6.0)
            win = 0
            for j in range(i+1, min(i+20, len(df))):
                if highs[j] >= sl:
                    break
                if lows[j] <= tp:
                    win = 1
                    break
            labels.append(win)
        else:
            labels.append(0)
            
    # Pad the end
    labels.extend([np.nan] * 20)
    df['Target'] = labels
    return df.dropna(subset=['Target'])

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
def train_all_swing_models():
    print("="*50)
    print(" TRAINING SWING AI MODELS (v5 PRO MAX)")
    print("="*50)
    
    feature_cols = ['Weekly_Bias', 'RSI', 'EMA9', 'EMA21', 'SMA50']
    
    for pair_name, ticker in PAIRS.items():
        print(f"Training {pair_name}...")
        try:
            t = yf.Ticker(ticker)
            df_1wk = t.history(period="10y", interval="1wk")
            df_1d  = t.history(period="5y", interval="1d")
            
            if len(df_1d) < 200:
                print(f"  Not enough data for {pair_name}")
                continue
                
            df_feat = build_swing_features(df_1d, df_1wk)
            df_labeled = label_swing_trades(df_feat)
            
            if df_labeled.empty:
                continue

            X = df_labeled[feature_cols]
            y = df_labeled['Target']
            
            # 80/20 Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                eval_metric='logloss',
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            
            # Save
            model_path = f"{MODELS_DIR}/swing_{pair_name}_model.json"
            model.save_model(model_path)
            print(f"  {pair_name} | Accuracy: {acc*100:.1f}% | Saved: {model_path}")
            
        except Exception as e:
            print(f"  Failed on {pair_name}: {e}")

if __name__ == "__main__":
    train_all_swing_models()
    print("\nTraining Complete! v5 PRO MAX brains are ready.")
