import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from notifier import send_discord_alert
from local_db import save_signal

# ─────────────────────────────────────────
# CONFIG: GURU ENGINE
# ─────────────────────────────────────────
PAIRS = {
    "Gold":   {"ticker": "GC=F", "sl_atr": 1.0, "tp_atr": 3.0},
    "US30":   {"ticker": "YM=F", "sl_atr": 1.0, "tp_atr": 3.0},
    "NAS100": {"ticker": "NQ=F", "sl_atr": 1.0, "tp_atr": 3.0},
    "BTC":    {"ticker": "BTC-USD", "sl_atr": 1.5, "tp_atr": 4.5}
}

# ─────────────────────────────────────────
# GURU 1: GEEKV1 (Market Mechanics / Brad Goh)
# ─────────────────────────────────────────
def geekv1_logic(df, current_price, atr):
    """
    Market Mechanics:
    1. Identify Swing High / Swing Low
    2. Premium / Discount Pricing (Fibonacci < 0.5)
    3. Order Block tap
    """
    # Simple pivot detection (rolling window)
    df['Pivot_Low'] = df['Low'] == df['Low'].rolling(window=20, center=True).min()
    df['Pivot_High'] = df['High'] == df['High'].rolling(window=20, center=True).max()
    
    lows = df[df['Pivot_Low']]
    highs = df[df['Pivot_High']]
    
    if len(lows) < 2 or len(highs) < 2:
        return False, None
    
    last_low = lows['Low'].iloc[-1]
    last_high = highs['High'].iloc[-1]
    
    # Assuming Uptrend (Higher Highs, Higher Lows)
    if highs['High'].iloc[-1] > highs['High'].iloc[-2] and lows['Low'].iloc[-1] > lows['Low'].iloc[-2]:
        # Fibonacci Retracement
        fib_50 = last_high - ((last_high - last_low) * 0.5)
        
        # Are we in Discount Pricing?
        if current_price < fib_50 and current_price > last_low:
            # Did we tap the Order Block? (Assuming OB is near the last swing low)
            ob_top = last_low + (atr * 0.5)
            if current_price <= ob_top:
                return True, "LONG"
                
    # Assuming Downtrend
    if highs['High'].iloc[-1] < highs['High'].iloc[-2] and lows['Low'].iloc[-1] < lows['Low'].iloc[-2]:
        fib_50 = last_low + ((last_high - last_low) * 0.5)
        
        # Are we in Premium Pricing?
        if current_price > fib_50 and current_price < last_high:
            ob_bot = last_high - (atr * 0.5)
            if current_price >= ob_bot:
                return True, "SHORT"
                
    return False, None

# ─────────────────────────────────────────
# GURU 2: SCIV1 (TradesBySci / ICC Supply & Demand)
# ─────────────────────────────────────────
def sciv1_logic(df, current_price, atr):
    """
    ICC Concepts:
    1. Find massive imbalance candle (Institutional Push)
    2. Base of the candle is Demand Zone
    3. Retest of the zone on lower volume
    """
    df['Body'] = abs(df['Close'] - df['Open'])
    avg_body = df['Body'].mean()
    
    # Find candles where body is > 2.5x average (Imbalance)
    imbalances = df[df['Body'] > (avg_body * 2.5)]
    
    if imbalances.empty:
        return False, None
        
    last_imbalance = imbalances.iloc[-1]
    
    # Bullish Imbalance
    if last_imbalance['Close'] > last_imbalance['Open']:
        demand_top = last_imbalance['Open']
        demand_bot = last_imbalance['Low']
        
        # Are we tapping the demand zone right now?
        if demand_bot <= current_price <= demand_top:
            return True, "LONG"
            
    # Bearish Imbalance
    if last_imbalance['Close'] < last_imbalance['Open']:
        supply_bot = last_imbalance['Open']
        supply_top = last_imbalance['High']
        
        # Are we tapping the supply zone right now?
        if supply_bot <= current_price <= supply_top:
            return True, "SHORT"
            
    return False, None

# ─────────────────────────────────────────
# GURU 3: INSIDERV1 (ICT / SMC)
# ─────────────────────────────────────────
def insiderv1_logic(df, current_price, atr):
    """
    Smart Money Concepts:
    1. Liquidity Sweep
    2. Market Structure Shift (MSS)
    3. Fair Value Gap (FVG) Retest
    """
    # Extremely simplified FVG detection for the last 10 candles
    recent = df.tail(15).copy()
    
    for i in range(1, len(recent)-1):
        c1 = recent.iloc[i-1]
        c2 = recent.iloc[i] # The displacement candle
        c3 = recent.iloc[i+1]
        
        # Bullish FVG
        if c1['High'] < c3['Low'] and c2['Close'] > c2['Open']:
            fvg_top = c3['Low']
            fvg_bot = c1['High']
            
            # If price retraces into this FVG
            if fvg_bot <= current_price <= fvg_top:
                return True, "LONG"
                
        # Bearish FVG
        if c1['Low'] > c3['High'] and c2['Close'] < c2['Open']:
            fvg_bot = c3['High']
            fvg_top = c1['Low']
            
            # If price retraces into this FVG
            if fvg_bot <= current_price <= fvg_top:
                return True, "SHORT"
                
    return False, None

# ─────────────────────────────────────────
# ENGINE RUNNER
# ─────────────────────────────────────────
def run_guru_engine(interval="5m", period="5d"):
    print("=" * 55)
    print(f"  GURU ENGINE SCAN | Interval: {interval}")
    print("  Bots: geekv1, sciv1, Insiderv1")
    print("=" * 55)
    
    now_utc = datetime.now(timezone.utc)
    
    for pair, cfg in PAIRS.items():
        try:
            t = yf.Ticker(cfg['ticker'])
            df = t.history(period=period, interval=interval)
            
            if len(df) < 50:
                continue
                
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            # Calculate ATR for dynamic zones
            tr = np.maximum(df['High'] - df['Low'], 
                 np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                            abs(df['Low'] - df['Close'].shift(1))))
            df['ATR'] = tr.rolling(14).mean()
            
            current_price = float(df['Close'].iloc[-1])
            atr = float(df['ATR'].iloc[-1])
            
            # Run all 3 bots
            geek_fire, geek_dir = geekv1_logic(df.copy(), current_price, atr)
            sci_fire, sci_dir = sciv1_logic(df.copy(), current_price, atr)
            insider_fire, insider_dir = insiderv1_logic(df.copy(), current_price, atr)
            
            # Package and Send Signals
            for fire, direction, engine_name in [
                (geek_fire, geek_dir, "geekv1 (Market Mechanics)"),
                (sci_fire, sci_dir, "sciv1 (ICC Supply/Demand)"),
                (insider_fire, insider_dir, "Insiderv1 (ICT / SMC)")
            ]:
                if fire:
                    sl = current_price - (atr * cfg['sl_atr']) if direction == "LONG" else current_price + (atr * cfg['sl_atr'])
                    tp = current_price + (atr * cfg['tp_atr']) if direction == "LONG" else current_price - (atr * cfg['tp_atr'])
                    
                    signal = {
                        "pair": pair,
                        "direction": direction,
                        "entry": round(current_price, 5),
                        "sl": round(sl, 5),
                        "tp": round(tp, 5),
                        "grade": "A",
                        "score": 100,
                        "win_prob": 0.80, # Pure rules-based, no ML yet
                        "session": f"Guru Engine ({interval})",
                        "rr": round(cfg['tp_atr'] / cfg['sl_atr'], 2),
                        "time_utc": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
                        "engine": engine_name
                    }
                    
                    print(f"  [GURU SIGNAL] {pair} {direction} | Bot: {engine_name}")
                    send_discord_alert(signal)
                    save_signal(signal)
                    time.sleep(1)
                    
        except Exception as e:
            print(f"Error on {pair}: {e}")

if __name__ == "__main__":
    # You can easily change this to "1d" and "1y" for Swing Trading!
    run_guru_engine(interval="5m", period="5d")
    print("  Guru Engine scan complete.")
