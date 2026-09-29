import yfinance as yf
import pandas as pd
import numpy as np

def check_ltf_confirmation(ticker, direction, interval="15m", period="3d"):
    """
    Checks the Lower Timeframe (LTF) for a Break of Structure (ChoCh) / Retest confirmation.
    Returns: (bool is_confirmed, str reason)
    """
    try:
        # 1. Download LTF data
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df.empty or len(df) < 20:
            return False, "Not enough LTF data"
            
        # 2. Identify Swing Highs and Swing Lows on LTF
        df['Pivot_Low'] = df['Low'] == df['Low'].rolling(window=5, center=True).min()
        df['Pivot_High'] = df['High'] == df['High'].rolling(window=5, center=True).max()
        
        lows = df[df['Pivot_Low']]['Low']
        highs = df[df['Pivot_High']]['High']
        
        if len(lows) < 2 or len(highs) < 2:
            return False, "No clear structure yet (Consolidation)"
            
        last_low = lows.iloc[-1]
        prev_low = lows.iloc[-2]
        last_high = highs.iloc[-1]
        prev_high = highs.iloc[-2]
        current_price = df['Close'].iloc[-1]

        # 3. Check for Confirmation
        if direction == "LONG":
            # We want to see a ChoCh (Change of Character) -> Breaking the previous swing high
            # AND a retest (Higher Low)
            
            # Did we break structure up? (New High > Prev High)
            bos_up = last_high > prev_high
            
            # Do we have a higher low? (Retest holding)
            higher_low = last_low > prev_low
            
            # Are we currently above the last low? (Not dumping)
            holding_level = current_price > last_low
            
            if not bos_up:
                return False, "Falling Knife: No LTF Break of Structure Up (ChoCh)"
            if not higher_low:
                return False, "Falling Knife: Still making Lower Lows on LTF"
            if not holding_level:
                return False, "Greedy Entry: Price is breaking the retest level"
                
            return True, "LTF Confirmed: Break & Retest Up"

        elif direction == "SHORT":
            # We want to see a ChoCh -> Breaking the previous swing low
            # AND a retest (Lower High)
            
            bos_down = last_low < prev_low
            lower_high = last_high < prev_high
            holding_level = current_price < last_high
            
            if not bos_down:
                return False, "Catching Rockets: No LTF Break of Structure Down"
            if not lower_high:
                return False, "Catching Rockets: Still making Higher Highs on LTF"
            if not holding_level:
                return False, "Greedy Entry: Price is breaking the retest level up"
                
            return True, "LTF Confirmed: Break & Retest Down"

    except Exception as e:
        return False, f"LTF Error: {e}"
