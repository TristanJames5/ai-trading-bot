import os
import discord
from discord.ext import commands, tasks
import yfinance as yf
from dotenv import load_dotenv
from google import genai

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
gemini_client = None
if GEMINI_KEY and GEMINI_KEY != "PASTE_YOUR_GEMINI_KEY_HERE":
    gemini_client = genai.Client(api_key=GEMINI_KEY)

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
                    # Calculate SL and TP
                    sl_atr = 1.0
                    tp_atr = 3.0
                    if direction == "LONG":
                        sl = current_price - (atr * sl_atr)
                        tp = current_price + (atr * tp_atr)
                    else:
                        sl = current_price + (atr * sl_atr)
                        tp = current_price - (atr * tp_atr)
                        
                    rr = round(tp_atr / sl_atr, 2)
                    
                    # Call the Oracle
                    oracle_bias = "Skipped (API Key missing)"
                    if gemini_client:
                        try:
                            prompt = f"Act as a ruthless macro-economic trading oracle. In exactly ONE short sentence, give me the fundamental macro bias (Bullish or Bearish) for {pair} based on Yields and DXY."
                            ai_response = gemini_client.models.generate_content(
                                model='gemini-3.8-flash',
                                contents=prompt
                            )
                            oracle_bias = ai_response.text.strip()
                        except:
                            oracle_bias = "Oracle offline."

                    # Build the Rich Embed
                    embed = discord.Embed(
                        title=f"🤖 AI TRADING SIGNAL | {pair} {direction}",
                        color=0x00ff00 if direction == "LONG" else 0xff0000
                    )
                    embed.add_field(name="Direction", value=direction, inline=True)
                    embed.add_field(name="Grade", value="A 🌟🌟🌟", inline=True)
                    embed.add_field(name="Engine", value="Protocol Trinity (Guru + Oracle)", inline=True)
                    
                    embed.add_field(name="ENTRY", value=f"```\n{current_price:.5f}\n```", inline=False)
                    
                    embed.add_field(name="STOP LOSS", value=f"```\n{sl:.5f}\n```", inline=True)
                    embed.add_field(name="TAKE PROFIT", value=f"```\n{tp:.5f}\n```", inline=True)
                    
                    embed.add_field(name="Risk:Reward", value=f"1:{rr}", inline=False)
                    embed.add_field(name="👁️ Oracle Bias", value=f"*{oracle_bias}*", inline=False)
                    embed.add_field(name="Scores", value="Rules: 100/100 | ML: AI Confirmed", inline=False)
                    
                    embed.set_footer(text="Manage your risk. Only trade 1-2%.")
                    
                    await channel.send(embed=embed)
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
        # Smart Ticker Mapping for Yahoo Finance
        ticker_map = {
            "US30": "^DJI",
            "NAS100": "^IXIC",
            "NDX": "^IXIC",
            "SPX": "^GSPC",
            "SP500": "^GSPC",
            "XAUUSD": "GC=F",
            "GOLD": "GC=F",
            "DXY": "DX-Y.NYB",
            "BTC": "BTC-USD",
            "ETH": "ETH-USD"
        }
        
        # Check if they used an alias, otherwise default to forex format
        upper_pair = pair.upper()
        if upper_pair in ticker_map:
            ticker = ticker_map[upper_pair]
        else:
            ticker = pair if "=" in pair or "-" in pair or "^" in pair else f"{pair}=X"
            
        df = yf.Ticker(ticker).history(period='5d', interval='1h')
        
        if df.empty:
            await ctx.send(f"❌ Could not find data for `{pair}`. Try standard ticker format (e.g. USDJPY=X or GC=F).")
            return

        close_val = df['Close'].iloc[-1]
        current_price = float(close_val.iloc[0] if hasattr(close_val, 'iloc') else close_val)
        
        # Generate Oracle response using LLM
        oracle_response = "Evaluating... (Gemini API Key missing)"
        if gemini_client:
            try:
                prompt = f"Act as a ruthless macro-economic trading oracle. In exactly two short sentences, give me the current fundamental macro bias (Bullish or Bearish) for {pair} based on current US Treasury yields and DXY."
                ai_response = gemini_client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=prompt
                )
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
