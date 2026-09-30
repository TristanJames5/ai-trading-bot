import os
import MetaTrader5 as mt5
from dotenv import load_dotenv

load_dotenv()

MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

def init_mt5():
    # Initialize with credentials and 60 second timeout
    if not mt5.initialize(path=TERMINAL_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, timeout=60000):
        print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return False
        
    acc = mt5.account_info()
    if acc is None or acc.login != MT5_LOGIN:
        print("Not logged into the correct account. Attempting login...")
        if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            print(f"MT5 login failed, error code: {mt5.last_error()}")
            return False
            
    return True

def map_symbol(pair_name):
    # Map the python bot pair names to MT5 base symbol names
    base = pair_name
    if pair_name == "Gold": base = "XAUUSD"
    elif pair_name == "BTC": base = "BTCUSD"
    
    # Try to find the exact symbol in MT5 (e.g. BTCUSDm)
    symbols = mt5.symbols_get(group=f"*{base}*")
    if symbols:
        # Return the first one that matches
        for s in symbols:
            if base in s.name:
                return s.name
        return symbols[0].name
    return base

def open_trade(pair_name, direction, risk_level, sl, tp):
    """Opens a market order. Returns None if daily loss limit is breached."""
    if not init_mt5(): return None
    
    # --- PROP FIRM DAILY LOSS GUARD ---
    # Block new trades if account has dropped > 3.5% from start of day equity
    # (Prop firm limit is 4%, we stop at 3.5% to leave a safety buffer)
    acc_check = mt5.account_info()
    if acc_check:
        balance     = acc_check.balance
        equity      = acc_check.equity
        daily_dd    = (balance - equity) / balance if balance > 0 else 0
        if daily_dd > 0.035:
            print(f"  [GUARD] Daily loss limit at {daily_dd:.1%} — NO NEW TRADES until tomorrow.")
            return None
    
    symbol = map_symbol(pair_name)
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"MT5 Symbol not found: {symbol}")
        return None
        
    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)
            
    order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if direction == "LONG" else mt5.symbol_info_tick(symbol).bid
    
    # -----------------------------------------
    # DYNAMIC LOT SIZE CALCULATION (0.5% RISK)
    # -----------------------------------------
    acc_info = mt5.account_info()
    if acc_info is None:
        print("Failed to get MT5 account info for lot sizing.")
        return None
        
    equity = acc_info.equity
    
    if risk_level == "Light (1%)": risk_pct = 0.0025    # 0.25% risk
    elif risk_level == "Normal (5%)": risk_pct = 0.005  # 0.50% risk ($25 on $5k)
    elif risk_level == "MAX (10%)": risk_pct = 0.01     # 1.00% risk ($50 on $5k)
    else: risk_pct = 0.005
    
    risk_amount_usd = equity * risk_pct
    
    tick_size = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value
    
    if tick_size == 0 or tick_value == 0:
        lot_size = symbol_info.volume_min
    else:
        sl_distance = abs(price - sl)
        sl_ticks = sl_distance / tick_size
        sl_value_1_lot = sl_ticks * tick_value
        
        if sl_value_1_lot > 0:
            lot_size = risk_amount_usd / sl_value_1_lot
        else:
            lot_size = symbol_info.volume_min
            
    # Round to allowed volume steps
    vol_step = symbol_info.volume_step
    lot_size = round(lot_size / vol_step) * vol_step
    
    # Clamp to min/max constraints
    if lot_size < symbol_info.volume_min: lot_size = symbol_info.volume_min
    if lot_size > symbol_info.volume_max: lot_size = symbol_info.volume_max
    
    lot_size = round(lot_size, 2)
    # -----------------------------------------
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot_size),
        "type": order_type,
        "price": price,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 20,
        "magic": 999999,
        "comment": "Sentinel AI",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"MT5 Order Send Failed, retcode={result.retcode}")
        return None
        
    print(f"MT5 Trade Opened: {symbol} {direction} (Ticket: {result.order})")
    return result.order

def close_trade(pair_name, direction):
    """Closes an existing order. We close all orders for this symbol with our magic number."""
    if not init_mt5(): return False
    
    symbol = map_symbol(pair_name)
    
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return False
        
    closed_any = False
    for pos in positions:
        if pos.magic == 999999: # Our bot's magic number
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": 999999,
                "comment": "Sentinel Early Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"MT5 Trade Closed Early: {symbol} (Ticket: {pos.ticket})")
                closed_any = True
            else:
                print(f"MT5 Close Order Failed, retcode={result.retcode}")
                
    return closed_any

def move_sl_to_be(pair_name, new_sl):
    """Moves Stop Loss to Break-Even."""
    if not init_mt5(): return False
    
    symbol = map_symbol(pair_name)
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return False
        
    moved_any = False
    for pos in positions:
        if pos.magic == 999999: # Our bot's magic number
            # Prevent moving SL backwards if it's already better than BE
            if pos.type == mt5.ORDER_TYPE_BUY and pos.sl >= new_sl: continue
            if pos.type == mt5.ORDER_TYPE_SELL and pos.sl > 0 and pos.sl <= new_sl: continue
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "sl": float(new_sl),
                "tp": pos.tp,
                "magic": 999999
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"MT5 SL moved to Break-Even: {symbol} (Ticket: {pos.ticket})")
                moved_any = True
            else:
                pass # Already at BE or minor error
                
    return moved_any
