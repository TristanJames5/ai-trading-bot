"""
TRAIN V5 — XGBoost Per-Pair Model Trainer
==========================================
Runs daily via GitHub Actions (midnight UTC).
Trains one XGBoost model per pair on rolling 180-day data.
Saves model files to /models/ folder (committed to GitHub automatically).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
PAIRS = {
    "US30":   "YM=F",
    "NAS100": "NQ=F",
    "Gold":   "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
}

SL_ATR = 1.2
TP_ATR = 3.0
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def build_features(df, df_1d):
    df = df.copy()

    # Trend
    df['EMA9']   = df['Close'].ewm(span=9).mean()
    df['EMA21']  = df['Close'].ewm(span=21).mean()
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SH50']   = df['High'].rolling(50).max()
    df['SL50']   = df['Low'].rolling(50).min()

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

    # FVG
    df['FVG_Bull'] = (df['Low'] > df['High'].shift(2)).astype(int)
    df['FVG_Bear'] = (df['High'] < df['Low'].shift(2)).astype(int)

    # OB
    body = abs(df['Close'] - df['Open'])
    df['OB_Bull'] = ((df['Close'].shift(1) < df['Open'].shift(1)) &
                     (df['Close'] > df['Open']) &
                     (body > df['ATR'])).astype(int)
    df['OB_Bear'] = ((df['Close'].shift(1) > df['Open'].shift(1)) &
                     (df['Close'] < df['Open']) &
                     (body > df['ATR'])).astype(int)

    # PDH/PDL
    daily_high = df['High'].resample('D').max()
    daily_low  = df['Low'].resample('D').min()
    df['PDH'] = daily_high.shift(1).reindex(df.index, method='ffill')
    df['PDL'] = daily_low.shift(1).reindex(df.index, method='ffill')
    df['Swept_High'] = ((df['High'] > df['PDH']) & (df['Close'] < df['PDH'])).astype(int)
    df['Swept_Low']  = ((df['Low']  < df['PDL']) & (df['Close'] > df['PDL'])).astype(int)

    # Session (encoded as number)
    def session_code(hour):
        if  7 <= hour <  9: return 3   # London kill zone — best
        if 13 <= hour < 16: return 2   # NY kill zone
        if  9 <= hour < 12: return 1   # London mid
        if  0 <= hour <  7: return -1  # Asia — avoid
        return 0
    df['Session'] = df.index.hour.map(session_code)

    # Daily bias (from daily EMA)
    df_1d['EMA9d']  = df_1d['Close'].ewm(span=9).mean()
    df_1d['EMA21d'] = df_1d['Close'].ewm(span=21).mean()
    df_1d['SMA50d'] = df_1d['Close'].rolling(50).mean()
    df_1d['Bias']   = np.where(
        (df_1d['EMA9d'] > df_1d['EMA21d']) & (df_1d['Close'] > df_1d['SMA50d']), 1,
        np.where((df_1d['EMA9d'] < df_1d['EMA21d']) & (df_1d['Close'] < df_1d['SMA50d']), -1, 0)
    )
    df_1d_bias = df_1d[['Bias']].copy()
    df_1d_bias.index = pd.to_datetime(df_1d_bias.index).normalize().tz_localize(None)
    df['Daily_Bias'] = df.index.normalize().map(df_1d_bias['Bias'].to_dict())
    df['Daily_Bias'] = df['Daily_Bias'].ffill().fillna(0)

    # Derived features
    df['Zone_Pct']    = (df['Close'] - df['SL50']) / (df['SH50'] - df['SL50'] + 1e-9)
    df['EMA_Dist']    = abs(df['Close'] - df['EMA21']) / (df['ATR'] + 1e-9)
    df['Price_SMA50'] = (df['Close'] - df['SMA50']) / (df['ATR'] + 1e-9)
    df['EMA_Cross']   = (df['EMA9'] > df['EMA21']).astype(int)
    df['RSI_norm']    = df['RSI'] / 100

    return df.dropna()


# ─────────────────────────────────────────
# LABEL GENERATION (did this candle lead to a win?)
# ─────────────────────────────────────────
def generate_labels(df, bias_col='Daily_Bias'):
    labels = []
    for i in range(len(df) - 50):
        row  = df.iloc[i]
        bias = int(row[bias_col])
        atr  = float(row['ATR'])
        if bias == 0 or atr == 0:
            labels.append(np.nan); continue

        entry = float(row['Close'])
        sl    = entry - atr * SL_ATR if bias == 1 else entry + atr * SL_ATR
        tp    = entry + atr * TP_ATR if bias == 1 else entry - atr * TP_ATR

        result = np.nan
        for j in range(i+1, min(i+50, len(df))):
            future = df.iloc[j]
            hi, lo = float(future['High']), float(future['Low'])
            if bias == 1:
                if hi >= tp: result = 1; break
                if lo <= sl: result = 0; break
            else:
                if lo <= tp: result = 1; break
                if hi >= sl: result = 0; break
        labels.append(result)

    labels += [np.nan] * 50
    return labels


# ─────────────────────────────────────────
# FEATURES USED FOR TRAINING
# ─────────────────────────────────────────
FEATURE_COLS = [
    'Daily_Bias', 'Session', 'Zone_Pct', 'EMA_Dist',
    'Price_SMA50', 'EMA_Cross', 'RSI_norm',
    'FVG_Bull', 'FVG_Bear', 'OB_Bull', 'OB_Bear',
    'Swept_High', 'Swept_Low'
]


# ─────────────────────────────────────────
# TRAIN + SAVE PER PAIR
# ─────────────────────────────────────────
def train_pair(pair_name, ticker):
    print(f"\n[{pair_name}] Downloading data...")
    try:
        df_1h = yf.download(ticker, period="730d", interval="1h",
                            progress=False, auto_adjust=True)
        df_1d = yf.download(ticker, period="2y", interval="1d",
                            progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  SKIP: {e}"); return

    for df in [df_1h, df_1d]:
        if df is None or df.empty:
            print("  SKIP: Empty data"); return
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    df_1h.index = pd.to_datetime(df_1h.index).tz_localize(None)
    df_1d.index = pd.to_datetime(df_1d.index).tz_localize(None)

    print(f"  Building features...")
    df_feat = build_features(df_1h, df_1d)

    print(f"  Generating labels...")
    df_feat['Label'] = generate_labels(df_feat)
    df_clean = df_feat.dropna(subset=['Label'] + FEATURE_COLS)
    df_clean = df_clean[df_clean['Daily_Bias'] != 0]   # Only directional bars

    if len(df_clean) < 100:
        print(f"  SKIP: Not enough labeled samples ({len(df_clean)})"); return

    X = df_clean[FEATURE_COLS]
    y = df_clean['Label'].astype(int)

    # Walk-forward split (70% train, 30% test — NO shuffling)
    split = int(len(X) * 0.70)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"  Training XGBoost ({len(X_train)} train, {len(X_test)} test)...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,           # Shallow = less overfit
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    overfit   = train_acc - test_acc

    print(f"  Train acc: {train_acc:.1%} | Test acc: {test_acc:.1%} | Overfit gap: {overfit:.1%}")

    if overfit > 0.15:
        print(f"  WARNING: Overfit gap > 15% — model may not generalize well")

    # Save model
    model_path = f"{MODELS_DIR}/{pair_name}_model.json"
    model.save_model(model_path)

    # Save metadata
    meta = {
        "pair":         pair_name,
        "ticker":       ticker,
        "trained_on":   datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "train_acc":    round(train_acc, 4),
        "test_acc":     round(test_acc, 4),
        "overfit_gap":  round(overfit, 4),
        "n_train":      len(X_train),
        "n_test":       len(X_test),
        "features":     FEATURE_COLS,
        "sl_atr":       SL_ATR,
        "tp_atr":       TP_ATR,
        "status":       "LIVE-READY" if test_acc >= 0.45 and overfit <= 0.15 else "CAUTION"
    }
    meta_path = f"{MODELS_DIR}/{pair_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: {model_path} [{meta['status']}]")
    return meta


# ─────────────────────────────────────────
# RUN ALL PAIRS
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  V5 DAILY TRAINER — XGBoost Per-Pair")
    print(f"  Run: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 55)

    results = []
    for pair, ticker in PAIRS.items():
        meta = train_pair(pair, ticker)
        if meta: results.append(meta)

    # Summary
    print("\n" + "=" * 55)
    print("  TRAINING SUMMARY")
    print("=" * 55)
    print(f"  {'Pair':<10} {'Test Acc':>9} {'Overfit':>9} {'Status':>12}")
    print("-" * 55)
    for m in results:
        print(f"  {m['pair']:<10} {m['test_acc']:>8.1%} {m['overfit_gap']:>+8.1%} {m['status']:>12}")
    print("=" * 55)
    print("  Models saved to /models/")
    print("  GitHub Actions will commit these files automatically.")
    print("=" * 55)
