import subprocess
import sys
import time

def run_all_bots():
    print("========================================")
    print("🚀 STARTING ALL AI TRADING BOTS...")
    print("========================================")
    
    # Start the Standard Bot
    print("-> Starting Standard Engine (live_engine.py)...")
    standard_bot = subprocess.Popen([sys.executable, "live_engine.py"])
    
    # Give it a second to boot up before starting the second one
    time.sleep(2)
    
    # Start the Sentinel Bot
    print("-> Starting APEX Sentinel Engine (sentinel_engine.py)...")
    sentinel_bot = subprocess.Popen([sys.executable, "sentinel_engine.py"])

    # Give it a second to boot up
    time.sleep(2)

    # Start the Guru Bot
    print("-> Starting The GURU Engine (guru_engine.py)...")
    guru_bot = subprocess.Popen([sys.executable, "guru_engine.py"])
    # Start the Discord Trinity Bot
    print("-> Starting Trinity Discord Bot (discord_bot.py)...")
    trinity_bot = subprocess.Popen([sys.executable, "discord_bot.py"])
    
    print("\n✅ All 4 bots (Standard, Sentinel, Guru, and Trinity) are now running simultaneously!")
    print("Press Ctrl+C at any time to stop them all.\n")
    
    try:
        # Keep the main script alive while the bots run in the background
        standard_bot.wait()
        sentinel_bot.wait()
        guru_bot.wait()
        trinity_bot.wait()
    except KeyboardInterrupt:
        print("\nStopping all bots...")
        standard_bot.terminate()
        sentinel_bot.terminate()
        guru_bot.terminate()
        trinity_bot.terminate()
        print("All bots stopped.")

if __name__ == "__main__":
    run_all_bots()
