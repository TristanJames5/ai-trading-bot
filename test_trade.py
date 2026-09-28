import os
import MetaTrader5 as mt5
from dotenv import load_dotenv

load_dotenv()

MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

def test_trade():
    print("Connecting to MT5...")
    # Connect to the newly installed MT5 with 60 second timeout
    if not mt5.initialize(path=TERMINAL_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER, timeout=60000):
        print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return
        
    acc = mt5.account_info()
    if acc is None or acc.login != MT5_LOGIN:
        print("Not logged into the correct account. Attempting login...")
        if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            print(f"MT5 login failed, error code: {mt5.last_error()}")
            return
            
    acc = mt5.account_info()
    print(f"Logged in successfully to {acc.company} (Server: {acc.server})!")
    print(f"Account Permissions -> Trade Allowed: {acc.trade_allowed}, Algo Trading Allowed by Broker: {acc.trade_expert}")
    
    if not acc.trade_allowed:
        print("ERROR: Trading is disabled! Are you logged in with an Investor Password instead of the Master Password?")
    if not acc.trade_expert:
        print(f"ERROR: Algo Trading is disabled by {acc.company} server for this account!")
    
    # FundedNext doesn't allow crypto on this account, let's use EURUSD
    symbols = mt5.symbols_get(group="*EURUSD*")
    if not symbols:
        print("Could not find any EURUSD symbols!")
        return
        
    eurusd_symbol = symbols[0].name
        
    print(f"Using symbol: {eurusd_symbol}")
    
    if not mt5.symbol_select(eurusd_symbol, True):
        print(f"Failed to select {eurusd_symbol}")
        return
        
    # Get current price
    tick = mt5.symbol_info_tick(eurusd_symbol)
    if not tick:
        print(f"Failed to get price for {eurusd_symbol}")
        return
        
    price = tick.ask
    print(f"Current Ask Price: {price}")
    
    # 0.01 lot test trade
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": eurusd_symbol,
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": price - 0.0050, # 50 pips SL
        "tp": price + 0.0050, # 50 pips TP
        "deviation": 20,
        "magic": 111111,
        "comment": "Test Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    print("Sending order...")
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order Failed, retcode={result.retcode}")
    else:
        print(f"SUCCESS! Trade Opened: {eurusd_symbol} BUY 0.01 (Ticket: {result.order})")
        print("Check your MT5 terminal to see it!")

if __name__ == "__main__":
    test_trade()
