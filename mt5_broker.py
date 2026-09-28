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
    import MetaTrader5 as mt5
    symbols = mt5.symbols_get(group=f"*{base}*")
    if symbols:
        # Return the first one that matches
        for s in symbols:
            if base in s.name:
                return s.name
        return symbols[0].name
    return base

def open_trade(pair_name, direction, risk_level, sl, tp):
    """Opens a market order."""
    if not init_mt5(): return None
    
    symbol = map_symbol(pair_name)
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"MT5 Symbol not found: {symbol}")
        return None
        
    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)
            
    order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if direction == "LONG" else mt5.symbol_info_tick(symbol).bid
    
    # Using minimum micro-lots (0.01) so you are only risking cents/dollars for testing
    lot_size = 0.01
    if risk_level == "Light (1%)": lot_size = 0.01
    elif risk_level == "Normal (5%)": lot_size = 0.02
    elif risk_level == "MAX (10%)": lot_size = 0.03
    
    # MT5 volume step check
    lot_size = round(lot_size, 2)
    
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
