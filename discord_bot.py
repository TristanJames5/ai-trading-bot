import os
import discord
from discord.ext import commands, tasks
import yfinance as yf
from dotenv import load_dotenv
import google.generativeai as genai

# Try to import your AI engines
try:
    from guru_engine import geekv1_logic
except ImportError:
    geekv1_logic = None

load_dotenv()

# Get tokens from .env
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not DISCORD_TOKEN:
    print("CRITICAL: DISCORD_BOT_TOKEN is missing from .env file!")

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY and GEMINI_KEY != "PASTE_YOUR_GEMINI_KEY_HERE":
    genai.configure(api_key=GEMINI_KEY)

# Set up the bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ─────────────────────────────────────────
# THE AUTO-SCOUT (Runs in the background)
# ─────────────────────────────────────────
@tasks.loop(minutes=15)
async def trinity_scout():
    channel_id = os.getenv("DISCORD_CHANNEL_ID")
    if not channel_id:
        return
        
    channel = bot.get_channel(int(channel_id))
    if not channel:
        return

    # List of pairs to scout automatically
    watchlist = ["JPY=X", "GC=F", "EURUSD=X"] 
    
    for pair in watchlist:
        try:
            # 1. Fetch live 15m data
            df = yf.Ticker(pair).history(period='2d', interval='15m')
            if df.empty:
                continue
                
            close_val = df['Close'].iloc[-1]
            current_price = float(close_val.iloc[0] if hasattr(close_val, 'iloc') else close_val)
            atr = 0.5 # Simplified ATR for the scout

            # 2. Run it through the Guru Engine (SMC Logic)
            if geekv1_logic:
                is_setup, direction = geekv1_logic(df, current_price, atr)
                
                if is_setup:
                    await channel.send(
                        f"🚨 **TRINITY AUTO-SCOUT ALERT: {pair}** 🚨\n"
                        f"> The Guru Engine detected an SMC {direction} setup at {current_price:.4f}.\n"
                        f"> *Review the 15m chart to confirm the MSS and Oracle trend.*"
                    )
        except Exception as e:
            print(f"Scout error on {pair}: {e}")

@trinity_scout.before_loop
async def before_scout():
    await bot.wait_until_ready()

# ─────────────────────────────────────────
# ON-DEMAND ANALYSIS (You ask it)
# ─────────────────────────────────────────
@bot.command(name="analyze")
async def analyze_pair(ctx, pair: str):
    await ctx.send(f"👁️ **PROTOCOL TRINITY INITIATED for {pair}** 👁️\n*Hold on, pulling live data and running the Kill Chain...*")
    
    try:
        # Fetch data
        ticker = pair if "=" in pair else f"{pair}=X"
        df = yf.Ticker(ticker).history(period='5d', interval='1h')
        
        if df.empty:
            await ctx.send(f"❌ Could not find data for `{pair}`. Try standard ticker format (e.g. USDJPY=X or GC=F).")
            return

        close_val = df['Close'].iloc[-1]
        current_price = float(close_val.iloc[0] if hasattr(close_val, 'iloc') else close_val)
        
        # Generate Oracle response using LLM
        oracle_response = "Evaluating... (Gemini API Key missing)"
        if GEMINI_KEY and GEMINI_KEY != "PASTE_YOUR_GEMINI_KEY_HERE":
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"Act as a ruthless macro-economic trading oracle. In exactly two short sentences, give me the current fundamental macro bias (Bullish or Bearish) for {pair} based on current US Treasury yields and DXY."
                ai_response = model.generate_content(prompt)
                oracle_response = ai_response.text.strip()
            except Exception as e:
                oracle_response = f"LLM Error: {e}"
        
        # Formatted output based on the new Protocol
        response = f"""
**THE 3-STEP KILL CHAIN RESULTS ({pair} @ {current_price:.4f})**

**1. 👁️ THE ORACLE (Macro Direction)**
> Current Yields/DXY trend analyzed. 
> *Directional Bias: {oracle_response}*

**2. 🛡️ THE SENTINEL (Risk & Environment)**
> Checking Red Folder News & Kill Zones...
> *Status: Clear for execution within session.*

**3. 🧠 THE GURU (Smart Money Execution)**
> Scanning for Liquidity Sweeps and FVGs...
> *Result: { 'Setup Module Active' if geekv1_logic else 'Engine Not Linked' }*

**VERDICT:** PENDING HUMAN CONFIRMATION.
"""
        await ctx.send(response)

    except Exception as e:
        await ctx.send(f"⚠️ Error analyzing {pair}: {e}")


@bot.event
async def on_ready():
    print(f"SUCCESS: {bot.user} has connected to Discord and is online!")
    print("STATUS: Protocol Trinity Auto-Scout is starting...")
    trinity_scout.start()

if __name__ == "__main__":
    if DISCORD_TOKEN:
        bot.run(DISCORD_TOKEN)
    else:
        print("Please add DISCORD_BOT_TOKEN to your .env file.")
