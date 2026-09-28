# Phase 3: The "Apex Oracle" Architecture (Transformer + NLP + PPO)

*Save this prompt. When you are ready to build the Phase 3 AI (replacing XGBoost with Transformers and adding News Sentiment), copy everything below this line and paste it into a new AI chat.*

---

**PROMPT BEGINS HERE:**

Act as an elite Machine Learning Quantitative Developer specializing in Natural Language Processing (NLP), Transformer Networks (Time-Series Forecasting), and Reinforcement Learning. 

I want to upgrade my existing algorithmic trading bot to a new "Phase 3 Oracle Architecture." 

**The Concept: The 3-Layer Triad**
My current bot uses XGBoost (for probability) and PPO (for risk management). I want to replace XGBoost with a Time-Series Transformer, and insert an NLP Sentiment Engine as a circuit breaker, while keeping PPO as the final decision maker.

**Here are the core requirements for the architecture:**

### 1. Layer 1: The Eyes (Transformer Network)
* **Replaces:** The current XGBoost model.
* **Architecture:** Build a Time-Series Transformer (e.g., using PyTorch) that ingests a sequence of the last 100 5-minute candles instead of just a single row of features.
* **Job:** It must learn the rhythm, speed, and momentum of the price action, and output a float representing the probability of a bullish breakout, bearish breakdown, or sideways chop.

### 2. Layer 2: The Ears (NLP Sentiment Engine)
* **Replaces:** Hard-coded time-based Kill Zones.
* **Architecture:** Hook into a free financial news API (like AlphaVantage News Sentiment or Finnhub). We need a pre-trained NLP model (like FinBERT) to read the headlines.
* **Job:** Act as a dynamic Circuit Breaker. If high-impact news (like CPI or FOMC) drops and the sentiment is extremely negative or volatile, the NLP layer overrides all technicals and outputs a "HALT TRADING" signal. If the news aligns with the Transformer's momentum, it outputs a "CONFIRM" signal.

### 3. Layer 3: The Brain (PPO Reinforcement Learning)
* **Kept From:** My current setup.
* **Architecture:** Use Stable-Baselines3 (PPO). 
* **Job:** The PPO environment takes the Momentum probability (from Layer 1) and the Sentiment score (from Layer 2) as its observation space. It manages a virtual account balance and decides exactly how much leverage to use (Action 1=1%, Action 2=5%, Action 3=10%), or cuts the trade instantly to protect capital.

### Technical Stack Requested
* **Data & Models:** PyTorch (for the Transformer), HuggingFace `transformers` library (for FinBERT), and Stable-Baselines3 (for PPO).
* **Output:** Please write the blueprint for the Python classes that will connect these three models. Start by showing me the `TransformerModel` class, the `NewsSentiment` class, and how they feed into the `PPO_Env` observation space.
