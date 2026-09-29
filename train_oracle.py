"""
ORACLE PHASE 3 — PYTORCH TRAINING ENGINE
=========================================
Trains the Time-Series Transformer on 100-candle sequences.
Runs across all 21 assets and saves models to /models/
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Import feature builders from existing v5 logic to ensure identical inputs
from train_v5 import PAIRS, build_features, FEATURE_COLS
from oracle_engine import OracleTransformer

SEQ_LEN = 100
BATCH_SIZE = 32
EPOCHS = 10
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        seq = self.X[idx : idx + self.seq_len]
        label = self.y[idx + self.seq_len - 1]
        return seq, label

def generate_3class_labels(df, lookahead=20, sl_atr=1.5, tp_atr=3.0):
    """
    0 = Bull (Hits TP above before SL below)
    1 = Bear (Hits TP below before SL above)
    2 = Sideways (Neither, or chops around)
    """
    labels = []
    for i in range(len(df) - lookahead):
        row = df.iloc[i]
        entry = float(row['Close'])
        atr = float(row['ATR'])
        
        if atr == 0 or np.isnan(atr):
            labels.append(2)
            continue
            
        bull_tp = entry + (atr * tp_atr)
        bull_sl = entry - (atr * sl_atr)
        
        bear_tp = entry - (atr * tp_atr)
        bear_sl = entry + (atr * sl_atr)
        
        result = 2 # Default Sideways
        for j in range(i+1, i+lookahead):
            future = df.iloc[j]
            hi, lo = float(future['High']), float(future['Low'])
            
            # Check Bull
            if hi >= bull_tp and lo > bull_sl:
                result = 0
                break
            # Check Bear
            if lo <= bear_tp and hi < bear_sl:
                result = 1
                break
            
            # If hit both SLs, it's sideways/chop
            if lo <= bull_sl and hi >= bear_sl:
                result = 2
                break
                
        labels.append(result)
        
    # Pad the end
    labels += [2] * lookahead
    return labels

def train_oracle_pair(pair_name, ticker):
    print(f"\n[{pair_name}] Downloading data for Oracle Training...")
    try:
        df_1h = yf.download(ticker, period="730d", interval="1h", progress=False)
        df_1d = yf.download(ticker, period="2y", interval="1d", progress=False)
    except Exception as e:
        print(f"  SKIP: {e}"); return
        
    for df in [df_1h, df_1d]:
        if df is None or df.empty: return
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    df_1h.index = pd.to_datetime(df_1h.index).tz_localize(None)
    df_1d.index = pd.to_datetime(df_1d.index).tz_localize(None)

    print(f"  Building features...")
    df_feat = build_features(df_1h, df_1d)
    
    print(f"  Generating labels...")
    df_feat['Label'] = generate_3class_labels(df_feat)
    df_clean = df_feat.dropna(subset=FEATURE_COLS)

    if len(df_clean) < SEQ_LEN + 100:
        print(f"  SKIP: Not enough samples"); return

    X = df_clean[FEATURE_COLS].values
    y = df_clean['Label'].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_dataset = SequenceDataset(X_train, y_train, SEQ_LEN)
    test_dataset = SequenceDataset(X_test, y_test, SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Initializing PyTorch Transformer (Device: {device})...")
    model = OracleTransformer(feature_dim=len(FEATURE_COLS), hidden_dim=64, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"  Training for {EPOCHS} Epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # Quick eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    acc = correct / total
    print(f"  => Final Test Accuracy: {acc:.1%}")

    # Save weights natively as zip so oracle_engine can read it
    model_path = f"{MODELS_DIR}/{pair_name}_oracle.zip"
    torch.save(model.state_dict(), model_path)
    print(f"  => Saved Phase 3 Weights: {model_path}")

if __name__ == "__main__":
    print("=========================================")
    print("  PHASE 3 ORACLE: MASS TRAINING INITIATED")
    print("=========================================")
    for pair, ticker in PAIRS.items():
        train_oracle_pair(pair, ticker)
    print("=========================================")
    print("ALL 21 ORACLE MODELS TRAINED AND SAVED!")
