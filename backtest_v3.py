"""
BACKTEST v3 — IMPROVED ENGINE
Fixes applied based on v2 results:
  1. ANTI-LATE-TREND: Penalize entries when price is extended from EMA (A+ was worse than B)
  2. PREMIUM/DISCOUNT scoring: Buy in discount, sell in premium
  3. FVG + OB detection as scoring bonuses (not hard gates)
  4. PARTIAL TP at 1:1 — half position closed, remaining runs to 1:3.3
  5. MULTI-ASSET: Run on US30, NAS100, Gold simultaneously
  6. Separate stats per asset + combined totals
"""

import yfinance as yf
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
ASSETS = {
    "US30":   "YM=F",
    "NAS100": "NQ=F",
    "GOLD":   "GC=F",
}

SCORE_A  = 70
SCORE_B  = 50
SL_ATR   = 1.2
TP1_ATR  = 1.2   # Partial TP at 1:1 (half position)
TP2_ATR  = 4.0   # Full TP at 1:3.3 (remaining half)
KILL_ZONES = [(7, 9), (13, 16)]


# ─────────────────────────────────────────
# INDICATOR ENGINE (manual, no pandas-ta)
# ─────────────────────────────────────────
def build_indicators(df):
    df = df.copy()
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA9']   = df['Close'].ewm(span=9).mean()
    df['EMA21']  = df['Close'].ewm(span=21).mean()

    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low']  - df['Close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(14).mean()

    # RSI
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # FVG (last 3 candles — bullish gap if low > high[2], bearish if high < low[2])
    df['FVG_Bull'] = df['Low'] > df['High'].shift(2)
    df['FVG_Bear'] = df['High'] < df['Low'].shift(2)

    # OB (impulsive reversal: prev candle opposite, current candle strong with ATR body)
    df['Body'] = abs(df['Close'] - df['Open'])
    df['OB_Bull'] = (df['Close'].shift(1) < df['Open'].shift(1)) & \
                    (df['Close'] > df['Open']) & \
                    (df['Body'] > df['ATR'])
    df['OB_Bear'] = (df['Close'].shift(1) > df['Open'].shift(1)) & \
                    (df['Close'] < df['Open']) & \
                    (df['Body'] > df['ATR'])

    return df.dropna()


# ─────────────────────────────────────────
# V3 SCORING — ANTI-LATE-TREND + SMC
# ─────────────────────────────────────────
def score_v3(row, bias, direction):
    score = 0

    # 1. Daily Bias (25 pts)
    if bias == direction:
        score += 25

    price  = row['Close']
    sma50  = row['SMA50']
    sma200 = row['SMA200']
    atr    = row['ATR']

    # 2. Trend via SMA (20 pts) — but PENALIZE if price is extended
    dist_from_sma50 = abs(price - sma50) / atr
    if direction == 1 and price > sma50 and price > sma200:
        # Penalize if too far from SMA (late entry)
        if dist_from_sma50 < 1.5:   score += 20   # Close to SMA = fresh trend
        elif dist_from_sma50 < 3.0: score += 10   # Okay distance
        else:                        score += 0    # Too extended, no points
    elif direction == -1 and price < sma50 and price < sma200:
        if dist_from_sma50 < 1.5:   score += 20
        elif dist_from_sma50 < 3.0: score += 10
        else:                        score += 0
    elif direction == 1  and price > sma50: score += 8   # Partial alignment
    elif direction == -1 and price < sma50: score += 8

    # 3. EMA9/21 momentum (10 pts)
    if direction == 1  and row['EMA9'] > row['EMA21']: score += 10
    if direction == -1 and row['EMA9'] < row['EMA21']: score += 10

    # 4. RSI zone (10 pts)
    rsi = row['RSI']
    if direction == 1  and 35 < rsi < 60: score += 10   # Not overbought
    if direction == -1 and 40 < rsi < 65: score += 10   # Not oversold

    # 5. Premium / Discount zone (15 pts) — NEW
    swing_high = row.get('SH50', price)
    swing_low  = row.get('SL50', price)
    rng = swing_high - swing_low
    if rng > 0:
        zone_pct = (price - swing_low) / rng
        if direction == 1  and zone_pct < 0.50: score += 15   # Discount = buy zone
        if direction == -1 and zone_pct > 0.50: score += 15   # Premium = sell zone
        if direction == 1  and zone_pct < 0.35: score += 5    # Deep discount bonus
        if direction == -1 and zone_pct > 0.65: score += 5    # Deep premium bonus

    # 6. FVG confirmation (10 pts) — NEW
    if direction == 1  and row.get('FVG_Bull', False): score += 10
    if direction == -1 and row.get('FVG_Bear', False): score += 10

    # 7. OB confirmation (10 pts) — NEW
    if direction == 1  and row.get('OB_Bull', False): score += 10
    if direction == -1 and row.get('OB_Bear', False): score += 10

    # 8. Kill zone (5 pts) — reduced from 15 (was over-rewarding late entries)
    hr = row.name.hour
    if any(lo <= hr <= hi for lo, hi in KILL_ZONES):
        score += 5

    return min(score, 100)


# ─────────────────────────────────────────
# PER-ASSET BACKTEST WITH PARTIAL TP
# ─────────────────────────────────────────
def run_backtest(ticker, name):
    print(f"\nLoading {name} ({ticker})...")
    try:
        df_1h = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
        df_1d = yf.download(ticker, period="2y",   interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  SKIP: Download failed — {e}"); return None

    for df in [df_1h, df_1d]:
        if df is None or df.empty:
            print(f"  SKIP: Empty data."); return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Daily bias
    df_1d['SMA50']  = df_1d['Close'].rolling(50).mean()
    df_1d['SMA200'] = df_1d['Close'].rolling(200).mean()
    df_1d['Bias']   = np.where(
        (df_1d['Close'] > df_1d['SMA50']) & (df_1d['Close'] > df_1d['SMA200']), 1,
        np.where((df_1d['Close'] < df_1d['SMA50']) & (df_1d['Close'] < df_1d['SMA200']), -1, 0)
    )

    # Align to 1H
    df_1d_bias = df_1d[['Bias']].copy()
    df_1d_bias.index = pd.to_datetime(df_1d_bias.index).normalize().tz_localize(None)
    df_1h.index = pd.to_datetime(df_1h.index).tz_localize(None)
    df_1h['Daily_Bias'] = df_1h.index.normalize().map(df_1d_bias['Bias'].to_dict())
    df_1h['Daily_Bias'] = df_1h['Daily_Bias'].ffill()

    # Indicators
    df_1h = build_indicators(df_1h)

    # Rolling swing high/low for Premium/Discount (50-bar window)
    df_1h['SH50'] = df_1h['High'].rolling(50).max()
    df_1h['SL50'] = df_1h['Low'].rolling(50).min()
    df_1h.dropna(inplace=True)

    print(f"  Rows to test: {len(df_1h)}")

    trades = []
    active = None
    attempt_a = attempt_b = blocked = 0

    for i in range(200, len(df_1h)):
        row   = df_1h.iloc[i]
        atr   = row['ATR']
        price = row['Close']
        bias  = int(row['Daily_Bias']) if not pd.isna(row['Daily_Bias']) else 0

        if pd.isna(atr) or atr == 0 or bias == 0:
            continue

        # Manage active trade (partial TP logic)
        if active:
            hi, lo = row['High'], row['Low']
            d = active['dir']

            # Partial TP hit (TP1)
            if not active.get('tp1_hit'):
                if d == 1 and hi >= active['tp1']:
                    active['tp1_hit'] = True
                    active['pnl_r'] += TP1_ATR / SL_ATR * 0.5  # half position
                elif d == -1 and lo <= active['tp1']:
                    active['tp1_hit'] = True
                    active['pnl_r'] += TP1_ATR / SL_ATR * 0.5

            # Full TP (remaining half)
            if d == 1 and hi >= active['tp2']:
                active['pnl_r'] += TP2_ATR / SL_ATR * 0.5
                active['result'] = 'W'
                trades.append(active); active = None; continue
            elif d == -1 and lo <= active['tp2']:
                active['pnl_r'] += TP2_ATR / SL_ATR * 0.5
                active['result'] = 'W'
                trades.append(active); active = None; continue

            # SL hit — if TP1 already hit, we only lose on remaining half
            if d == 1 and lo <= active['sl']:
                if active.get('tp1_hit'):
                    active['pnl_r'] -= 1.0 * 0.5   # half-size loss
                    active['result'] = 'BE+'         # Better than loss
                else:
                    active['pnl_r'] -= 1.0
                    active['result'] = 'L'
                trades.append(active); active = None; continue
            elif d == -1 and hi >= active['sl']:
                if active.get('tp1_hit'):
                    active['pnl_r'] -= 1.0 * 0.5
                    active['result'] = 'BE+'
                else:
                    active['pnl_r'] -= 1.0
                    active['result'] = 'L'
                trades.append(active); active = None; continue
            continue

        # Scoring
        score = score_v3(row, bias, bias)
        if   score >= SCORE_A: attempt_a += 1
        elif score >= SCORE_B: attempt_b += 1
        else: blocked += 1; continue

        grade = "A+" if score >= SCORE_A else "B"

        if bias == 1:
            entry  = price
            sl_val = entry - atr * SL_ATR
            tp1    = entry + atr * TP1_ATR
            tp2    = entry + atr * TP2_ATR
        else:
            entry  = price
            sl_val = entry + atr * SL_ATR
            tp1    = entry - atr * TP1_ATR
            tp2    = entry - atr * TP2_ATR

        active = {
            'time': row.name, 'dir': bias, 'entry': entry,
            'sl': sl_val, 'tp1': tp1, 'tp2': tp2,
            'grade': grade, 'score': score,
            'pnl_r': 0.0, 'result': None, 'tp1_hit': False
        }

    # Results
    wins   = [t for t in trades if t['result'] in ('W', 'BE+')]
    losses = [t for t in trades if t['result'] == 'L']
    total  = len(trades)
    net_r  = sum(t['pnl_r'] for t in trades)

    if total == 0:
        print(f"  No trades."); return None

    winrate = len(wins) / total * 100
    pf_num  = sum(t['pnl_r'] for t in trades if t['pnl_r'] > 0)
    pf_den  = abs(sum(t['pnl_r'] for t in trades if t['pnl_r'] < 0))
    pf      = pf_num / max(pf_den, 0.001)

    a_trades = [t for t in trades if t['grade'] == 'A+']
    b_trades = [t for t in trades if t['grade'] == 'B']
    a_wr = len([t for t in a_trades if t['result'] in ('W','BE+')]) / max(len(a_trades),1) * 100
    b_wr = len([t for t in b_trades if t['result'] in ('W','BE+')]) / max(len(b_trades),1) * 100

    return {
        'name': name, 'total': total, 'wins': len(wins), 'losses': len(losses),
        'winrate': winrate, 'net_r': net_r, 'pf': pf,
        'signals_week': total / (730/7),
        'a_wr': a_wr, 'b_wr': b_wr,
        'attempt_a': attempt_a, 'attempt_b': attempt_b, 'blocked': blocked
    }


# ─────────────────────────────────────────
# RUN ALL ASSETS
# ─────────────────────────────────────────
print("=" * 60)
print("  BACKTEST v3 — Weighted SMC Engine (Multi-Asset)")
print("=" * 60)

all_results = []
for name, ticker in ASSETS.items():
    r = run_backtest(ticker, name)
    if r:
        all_results.append(r)

# Print individual results
print("\n" + "=" * 60)
for r in all_results:
    print(f"\n  [{r['name']}]")
    print(f"  Trades: {r['total']}  |  WR: {r['winrate']:.1f}%  |  Net R: {r['net_r']:+.1f}R  |  PF: {r['pf']:.2f}")
    print(f"  Signals/week: {r['signals_week']:.1f}  |  A+ WR: {r['a_wr']:.1f}%  |  B WR: {r['b_wr']:.1f}%")

# Combined totals
if all_results:
    total_trades = sum(r['total'] for r in all_results)
    total_wins   = sum(r['wins']  for r in all_results)
    total_net_r  = sum(r['net_r'] for r in all_results)
    combined_wr  = total_wins / total_trades * 100
    signals_wk   = sum(r['signals_week'] for r in all_results)

    print("\n" + "=" * 60)
    print("  COMBINED (US30 + NAS100 + GOLD)")
    print("=" * 60)
    print(f"  Total Trades    : {total_trades}")
    print(f"  Combined WR     : {combined_wr:.1f}%")
    print(f"  Net R (2yr)     : {total_net_r:+.1f}R  (+{total_net_r:.1f}% @ 1% risk)")
    print(f"  Signals/week    : {signals_wk:.1f}  (across all 3 assets)")
    print("=" * 60)

    # Verdict
    print("\n  VERDICT:")
    if combined_wr >= 40 and signals_wk >= 8:
        print("  [PASS] SIGNIFICANT IMPROVEMENT -- System is tradeable")
    elif combined_wr >= 35:
        print("  [WARN] MARGINAL IMPROVEMENT -- Direction is right, needs tuning")
    else:
        print("  [FAIL] NOT ENOUGH IMPROVEMENT -- Need deeper rethink")
    print()
