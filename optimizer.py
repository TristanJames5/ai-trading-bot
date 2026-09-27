"""
PER-PAIR STRATEGY OPTIMIZER v1
================================
For each asset, this script runs a grid search over key parameters
to find the unique optimal strategy for that specific pair.

Parameters optimized per pair:
  - SL multiplier (ATR)
  - TP multiplier (ATR)
  - RSI buy/sell zones
  - Score threshold (how strict the entry filter is)
  - Trend distance tolerance (how close to SMA)

Validation: Walk-forward split (70% train / 30% test)
Goal: Maximize Profit Factor on unseen (test) data
      while keeping WR >= 45% and RRR >= 1.5
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product

# ─────────────────────────────────────────
# ASSETS TO OPTIMIZE
# ─────────────────────────────────────────
ASSETS = {
    "US30":        "YM=F",
    "NAS100":      "NQ=F",
    "Gold":        "GC=F",
    "EUR/USD":     "EURUSD=X",
    "GBP/USD":     "GBPUSD=X",
    "BTC":         "BTC-USD",
}

# ─────────────────────────────────────────
# GRID SEARCH PARAMETER SPACE
# ─────────────────────────────────────────
GRID = {
    "sl_atr":       [0.8, 1.0, 1.2, 1.5],          # SL tightness
    "tp_atr":       [1.5, 2.0, 2.5, 3.0, 4.0],     # TP target
    "score_thresh": [45, 55, 65],                    # Entry quality gate
    "rsi_low":      [30, 35, 40],                    # RSI buy zone upper bound
    "rsi_high":     [60, 65, 70],                    # RSI sell zone lower bound
    "sma_dist_max": [1.5, 2.5, 4.0],                # Max ATR distance from SMA
}

# ─────────────────────────────────────────
# INDICATORS (manual, no pandas-ta)
# ─────────────────────────────────────────
def add_indicators(df):
    df = df.copy()
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA9']   = df['Close'].ewm(span=9).mean()
    df['EMA21']  = df['Close'].ewm(span=21).mean()
    df['SH50']   = df['High'].rolling(50).max()
    df['SL50']   = df['Low'].rolling(50).min()

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

    # FVG
    df['FVG_Bull'] = df['Low'] > df['High'].shift(2)
    df['FVG_Bear'] = df['High'] < df['Low'].shift(2)

    return df.dropna()


# ─────────────────────────────────────────
# PARAMETERIZED SCORER
# ─────────────────────────────────────────
def score_entry(row, bias, direction, params):
    score = 0
    price  = row['Close']
    atr    = row['ATR']
    sma50  = row['SMA50']
    sma200 = row['SMA200']
    rsi    = row['RSI']

    if bias == direction: score += 25

    dist = abs(price - sma50) / (atr + 1e-9)
    if direction == 1 and price > sma50 and price > sma200:
        if dist < params['sma_dist_max']:   score += 20
        else:                               score += 0
    elif direction == -1 and price < sma50 and price < sma200:
        if dist < params['sma_dist_max']:   score += 20
        else:                               score += 0
    elif direction == 1  and price > sma50: score += 8
    elif direction == -1 and price < sma50: score += 8

    if direction == 1  and row['EMA9'] > row['EMA21']: score += 10
    if direction == -1 and row['EMA9'] < row['EMA21']: score += 10

    # RSI zone check
    if direction == 1  and rsi < params['rsi_low']:  score += 15
    if direction == -1 and rsi > params['rsi_high']: score += 15

    # Premium / Discount
    sh, sl = row['SH50'], row['SL50']
    rng = sh - sl
    if rng > 0:
        zone_pct = (price - sl) / rng
        if direction == 1  and zone_pct < 0.50: score += 15
        if direction == -1 and zone_pct > 0.50: score += 15

    # FVG bonus
    if direction == 1  and row.get('FVG_Bull', False): score += 10
    if direction == -1 and row.get('FVG_Bear', False): score += 10

    # Kill zone
    hr = row.name.hour if hasattr(row.name, 'hour') else 0
    if 7 <= hr <= 9 or 13 <= hr <= 16: score += 5

    return min(score, 100)


# ─────────────────────────────────────────
# SINGLE BACKTEST RUN (for a given param set)
# ─────────────────────────────────────────
def run_single(df, daily_bias, params):
    trades  = []
    active  = None

    for i in range(200, len(df)):
        row   = df.iloc[i]
        atr   = row['ATR']
        price = row['Close']
        date  = row.name.normalize() if hasattr(row.name, 'normalize') else row.name

        bias_val = daily_bias.get(date, 0)
        if pd.isna(bias_val) or bias_val == 0 or pd.isna(atr) or atr == 0:
            continue

        bias = int(bias_val)

        # Manage active trade
        if active:
            hi, lo = row['High'], row['Low']
            d = active['dir']
            if d == 1:
                if hi >= active['tp']:
                    trades.append({**active, 'pnl': params['tp_atr'] / params['sl_atr']})
                    active = None; continue
                if lo <= active['sl']:
                    trades.append({**active, 'pnl': -1.0})
                    active = None; continue
            else:
                if lo <= active['tp']:
                    trades.append({**active, 'pnl': params['tp_atr'] / params['sl_atr']})
                    active = None; continue
                if hi >= active['sl']:
                    trades.append({**active, 'pnl': -1.0})
                    active = None; continue
            continue

        score = score_entry(row, bias, bias, params)
        if score < params['score_thresh']:
            continue

        if bias == 1:
            entry  = price
            sl_val = entry - atr * params['sl_atr']
            tp_val = entry + atr * params['tp_atr']
        else:
            entry  = price
            sl_val = entry + atr * params['sl_atr']
            tp_val = entry - atr * params['tp_atr']

        active = {'dir': bias, 'entry': entry, 'sl': sl_val, 'tp': tp_val}

    if not trades:
        return None

    wins   = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    total  = len(trades)
    wr     = len(wins) / total
    net_r  = sum(t['pnl'] for t in trades)
    pf_n   = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    pf_d   = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf     = pf_n / max(pf_d, 0.001)
    rr     = params['tp_atr'] / params['sl_atr']

    return {'total': total, 'wr': wr, 'net_r': net_r, 'pf': pf, 'rr': rr}


# ─────────────────────────────────────────
# PER-PAIR OPTIMIZER
# ─────────────────────────────────────────
def optimize_pair(name, ticker):
    print(f"\n  Downloading {name} ({ticker})...")
    try:
        df_1h = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
        df_1d = yf.download(ticker, period="2y",   interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"    SKIP: {e}"); return None

    for df in [df_1h, df_1d]:
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Daily bias
    df_1d['SMA50']  = df_1d['Close'].rolling(50).mean()
    df_1d['SMA200'] = df_1d['Close'].rolling(200).mean()
    df_1d['Bias']   = np.where(
        (df_1d['Close'] > df_1d['SMA50']) & (df_1d['Close'] > df_1d['SMA200']), 1,
        np.where((df_1d['Close'] < df_1d['SMA50']) & (df_1d['Close'] < df_1d['SMA200']), -1, 0)
    )
    df_1d_bias = df_1d[['Bias']].copy()
    df_1d_bias.index = pd.to_datetime(df_1d_bias.index).normalize().tz_localize(None)

    df_1h = add_indicators(df_1h)
    df_1h.index = pd.to_datetime(df_1h.index).tz_localize(None)
    bias_map = df_1d_bias['Bias'].to_dict()

    # Walk-forward split: 70% train, 30% test
    split = int(len(df_1h) * 0.70)
    df_train = df_1h.iloc[:split].copy()
    df_test  = df_1h.iloc[split:].copy()
    print(f"    Train: {len(df_train)} bars | Test: {len(df_test)} bars")

    # Grid search on training data
    best_params = None
    best_pf     = 0
    total_combinations = (
        len(GRID['sl_atr']) * len(GRID['tp_atr']) *
        len(GRID['score_thresh']) * len(GRID['rsi_low']) *
        len(GRID['rsi_high']) * len(GRID['sma_dist_max'])
    )
    print(f"    Testing {total_combinations} parameter combinations...")

    for sl, tp, sc, rl, rh, sdm in product(
        GRID['sl_atr'], GRID['tp_atr'],
        GRID['score_thresh'], GRID['rsi_low'],
        GRID['rsi_high'], GRID['sma_dist_max']
    ):
        params = {
            'sl_atr': sl, 'tp_atr': tp,
            'score_thresh': sc,
            'rsi_low': rl, 'rsi_high': rh,
            'sma_dist_max': sdm
        }
        result = run_single(df_train, bias_map, params)
        if result is None: continue
        # Only consider if WR >= 40% and RRR >= 1.5 and min 20 trades
        if result['wr'] >= 0.40 and result['rr'] >= 1.5 and result['total'] >= 20:
            if result['pf'] > best_pf:
                best_pf     = result['pf']
                best_params = params
                best_train  = result

    if best_params is None:
        print(f"    No valid params found for {name}")
        return None

    # Validate on unseen test data
    test_result = run_single(df_test, bias_map, best_params)

    return {
        'name':        name,
        'best_params': best_params,
        'train':       best_train,
        'test':        test_result,
    }


# ─────────────────────────────────────────
# RUN OPTIMIZER
# ─────────────────────────────────────────
print("=" * 65)
print("  PER-PAIR STRATEGY OPTIMIZER")
print("  Walk-Forward Validation (70/30 split)")
print("  Goal: Max Profit Factor | Min WR 40% | Min RRR 1.5")
print("=" * 65)

pair_results = []
for name, ticker in ASSETS.items():
    r = optimize_pair(name, ticker)
    if r: pair_results.append(r)

# ─────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  OPTIMIZED STRATEGY REPORT (Per Pair)")
print("=" * 65)
print(f"  {'Pair':<12} | {'WR Train':>8} | {'WR Test':>8} | {'RRR':>5} | {'PF Test':>7} | {'SL':>4} | {'TP':>4} | {'Score':>5}")
print("-" * 65)

for r in pair_results:
    p = r['best_params']
    tr = r['train']
    te = r['test']

    wr_train = f"{tr['wr']*100:.1f}%"
    wr_test  = f"{te['wr']*100:.1f}%" if te else "N/A"
    pf_test  = f"{te['pf']:.2f}"      if te else "N/A"
    rr       = f"1:{p['tp_atr']/p['sl_atr']:.1f}"

    # Verdict flag
    if te and te['wr'] >= 0.45 and te['pf'] >= 1.3:
        flag = "[LIVE-READY]"
    elif te and te['wr'] >= 0.40:
        flag = "[MARGINAL]  "
    else:
        flag = "[OVERFIT?]  "

    print(f"  {r['name']:<12} | {wr_train:>8} | {wr_test:>8} | {rr:>5} | {pf_test:>7} | {p['sl_atr']:>4} | {p['tp_atr']:>4} | {p['score_thresh']:>5}")
    print(f"    {flag} RSI zone: <{p['rsi_low']} / >{p['rsi_high']}  |  SMA dist max: {p['sma_dist_max']} ATR")
    print()

print("=" * 65)
print("  [LIVE-READY] = WR >= 45% and PF >= 1.3 on unseen data")
print("  [MARGINAL]   = Profitable but needs more validation")
print("  [OVERFIT?]   = Train performance didn't carry to test")
print("=" * 65)
