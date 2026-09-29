import subprocess
import sys
import time
import schedule
import threading

def weekend_workout():
    print("========================================")
    print("🧠 INITIATING WEEKEND ML WORKOUT 🧠")
    print("========================================")
    print("The markets are closed. Training engines on the latest data...")
    
    try:
        # Run Sentinel Training
        print("-> Retraining Sentinel Engine (v5)...")
        subprocess.run([sys.executable, "train_v5.py"])
        
        # Run Guru Training
        print("-> Retraining Guru Engine...")
        subprocess.run([sys.executable, "train_guru.py"])
        
        print("✅ Weekend Workout Complete! Models have been permanently upgraded.")
    except Exception as e:
        print(f"❌ Workout failed: {e}")

def scheduler_thread():
    # Schedule the workout for Saturday at 2:00 AM
    schedule.every().saturday.at("02:00").do(weekend_workout)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

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
    
    # Start the Continuous Learning Scheduler
    print("-> Starting Continuous Learning Scheduler (Saturdays @ 2AM)...")
    trainer_thread = threading.Thread(target=scheduler_thread, daemon=True)
    trainer_thread.start()
    
    print("\n✅ All 4 bots (Standard, Sentinel, Guru, and Trinity) are now running simultaneously!")
    print("✅ Continuous Learning is ACTIVE.")
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
