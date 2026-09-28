# Phase 5: The "Black Swan" Hunter Architecture 

*Save this prompt. When you are ready to build a highly specialized AI designed exclusively to trade market crashes, panic, and massive news spikes, copy everything below this line and paste it into a new AI chat.*

---

**PROMPT BEGINS HERE:**

Act as an elite Machine Learning Quantitative Developer specializing in High-Frequency Trading (HFT) and Natural Language Processing (NLP). 

I want to build a highly specialized bot called the "Black Swan Hunter". This bot does not day-trade. It sleeps 99% of the year and only wakes up to trade massive volatility events, flash crashes, and breaking news.

**The Concept: The Chaos Triad**
Instead of Reinforcement Learning, we are combining **Natural Language Processing (NLP)**, **Rule-Based Hard Logic**, and **Time-Series Transformers**. 

**Here are the core requirements for the architecture:**

### 1. Layer 1: The Trigger (NLP Sentiment Engine)
* **Architecture:** Hook into a real-time Twitter/X API, Bloomberg, and ForexFactory RSS feeds using a pre-trained NLP model (like FinBERT or a custom Llama3 model).
* **Job:** It constantly scans the internet for absolute panic or euphoria. It is looking for keywords associated with Black Swan events (e.g., "Bankrupt", "Emergency Rate Cut", "Hacked", "War"). If it detects a massive spike in global fear/greed, it wakes up the rest of the bot.

### 2. Layer 2: The Filter (Rule-Based Hard Logic)
* **Architecture:** Strict, mathematically hard-coded rules analyzing the VIX (Volatility Index), ATR (Average True Range), and Tick Volume.
* **Job:** Prevents the bot from getting tricked by "fake news". If Layer 1 screams "CRASH!", Layer 2 checks the actual market data. If the spread hasn't widened and the volume hasn't spiked 10x above the moving average, Layer 2 blocks the trade. It demands mathematical proof of chaos.

### 3. Layer 3: The Execution (Transformer Network)
* **Architecture:** A fast, lightweight Transformer model (PyTorch) trained exclusively on 1-second and tick-chart data during historical crashes (like the 2020 COVID crash or the 2010 Flash Crash).
* **Job:** Once activated and approved by Layers 1 & 2, the Transformer drops into the 1-second timeframe. It reads the micro-momentum of the order flow to execute the perfect short/long snipe in the middle of the chaos, riding the massive liquidity grab and exiting before the market halts or bounces.

### Technical Stack Requested
* **Data & Models:** Python, Tweepy (Twitter API), HuggingFace (FinBERT), and PyTorch (for the Tick-Data Transformer).
* **Output:** Please write the blueprint for the Python classes that will connect these three layers. Start by showing me the `NewsTrigger` class, the `VolatilityFilter` class, and how they activate the `FlashCrashTransformer`.
