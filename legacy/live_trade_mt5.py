import time
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from stable_baselines3 import PPO

from features.make_features import compute_features

# --- CONFIG ---
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
VOLUME = 0.01  # Minimum lot size
DEVIATION = 20
MODEL_PATH = "train/ppo_xauusd_latest.zip"
WINDOW = 64

# Mapping: 0=Flat, 1=Long
# (If we had Short, it would be mapped here too)

def get_market_data(symbol, n=500):
    """Fetch recent candles from MT5"""
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n)
    if rates is None:
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df

def execute_trade(action, current_pos_type):
    """
    Execute trade based on model action vs current position.
    action: 0 (Flat), 1 (Long)
    current_pos_type: 0 (Flat), 1 (Long)
    """
    
    # Logic:
    # If Action is LONG (1) and we are FLAT (0) -> BUY
    # If Action is FLAT (0) and we are LONG (1) -> CLOSE BUY
    # If Action == Current -> Do nothing
    
    if action == current_pos_type:
        return
    
    # Close existing position if any
    if current_pos_type == 1: # We are Long, need to Close
        print("🔻 Closing Long Position...")
        close_position(mt5.POSITION_TYPE_BUY)
        
    # Open new position
    if action == 1: # We want to be Long
        print("🟢 Opening Long Position...")
        open_order(mt5.ORDER_TYPE_BUY)

def open_order(order_type):
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": VOLUME,
        "type": order_type,
        "price": price,
        "deviation": DEVIATION,
        "magic": 234000,
        "comment": "RL_Agent_v1",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    print(f"Order Send Result: {result.comment if result else 'Failed'}")

def close_position(position_type):
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions:
        for pos in positions:
            if pos.type == position_type:
                tick = mt5.symbol_info_tick(SYMBOL)
                price = tick.bid if position_type == mt5.ORDER_TYPE_BUY else tick.ask
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if position_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": DEVIATION,
                    "magic": 234000,
                    "comment": "RL_Agent_Close",
                }
                mt5.order_send(request)

def get_current_position_type():
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return 0 # Flat
    
    # Simplified: Assume only 1 position at a time for this magic number
    for pos in positions:
        if pos.magic == 234000:
            if pos.type == mt5.POSITION_TYPE_BUY:
                return 1
            # If we supported short, we'd check SELL here
            
    return 0

def main():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return

    print(f"✅ Connected to MT5: {mt5.terminal_info().name}")
    
    # Load Model
    model = PPO.load(MODEL_PATH)
    print(f"🧠 Model loaded: {MODEL_PATH}")

    print("🚀 Starting Live Trading Loop (Ctrl+C to stop)...")
    
    try:
        while True:
            # 1. Get Data
            df = get_market_data(SYMBOL, n=200) # Fetch enough for MA(50) + Window(64)
            if df is None:
                print("❌ Failed to fetch data, retrying...")
                time.sleep(10)
                continue
            
            # 2. Compute Features
            # We need to make sure we have the latest completed candle or current forming one?
            # Standard RL usually trades on 'Close' of previous bar.
            # Let's take the last 'Window' rows.
            
            _, feats, _ = compute_features(df)
            
            # Get observation (last 64 steps)
            # feats shape: (N, F)
            if len(feats) < WINDOW:
                print("⏳ Not enough data yet...")
                time.sleep(10)
                continue
                
            obs_features = feats[-WINDOW:] # (64, F)
            
            # Get current position state for Observation
            current_pos = get_current_position_type()
            
            # Construct Observation: [features_flat, pos]
            obs = np.concatenate([obs_features.reshape(-1), np.array([current_pos], dtype=np.float32)])
            
            # 3. Predict Action
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | Pos: {current_pos} | Action: {action} ({'Long' if action==1 else 'Flat'})")
            
            # 4. Execute
            execute_trade(action, current_pos)
            
            # Sleep until next check (e.g., every 10 seconds or 1 minute)
            # For H1 trading, we don't need to check constantly, but good to check reasonably often.
            time.sleep(10) 

    except KeyboardInterrupt:
        print("\n🛑 Stopping Agent...")
        mt5.shutdown()

if __name__ == "__main__":
    main()
