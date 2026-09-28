# Phase 6: The "Hive-Mind" Swarm Architecture

*Save this prompt. When you are ready to build a Multi-Agent Swarm system that controls hundreds of micro-bots simultaneously, copy everything below this line and paste it into a new AI chat.*

---

**PROMPT BEGINS HERE:**

Act as an elite Machine Learning Quantitative Developer specializing in Multi-Agent Systems, Ensemble Machine Learning, and Genetic Algorithms. 

I want to build an advanced algorithmic trading architecture called the "Hive-Mind". Instead of relying on one single "God Bot" to trade the market, I want to deploy a swarm of 1,000 hyper-specialized "micro-bots" that vote on market direction and share a single master bankroll.

**The Concept: Swarm Intelligence**
We are combining **Genetic Algorithms**, **Transformers (Time-Series)**, and **NLP (Natural Language Sentiment)**.

**Here are the core requirements for the architecture:**

### 1. Layer 1: The Global Observers (Transformers + NLP)
* **Architecture:** A central "brain" running a Time-Series Transformer (analyzing price correlation across 20 different assets) and an NLP model (reading global news).
* **Job:** This layer does not trade. It acts as the "Weather Station." It broadcasts the current market climate (e.g., "High Volatility, Bearish Sentiment, USD Strength") down to the swarm of micro-bots.

### 2. Layer 2: The Micro-Bot Swarm (Genetic Algorithms)
* **Architecture:** An array of 1,000 extremely lightweight, genetically evolved mini-bots. 
* **Job:** Every single bot is highly specialized. For example, Bot #42 ONLY knows how to trade EURUSD during the London Session when the NLP detects negative European news. Bot #899 ONLY buys Gold when the Transformer detects a sudden drop in Bitcoin. 
* **Mechanic:** They receive the "Weather Report" from Layer 1. If the current market matches their exact genetic specialty, they vote "BUY" or "SELL".

### 3. Layer 3: The Hive-Mind Execution (The Master Controller)
* **Architecture:** A dynamic Ensemble allocator.
* **Job:** It counts the votes from the 1,000 micro-bots. If 800 bots suddenly scream "SELL EURUSD", the Hive-Mind executes a massive Short position. 
* **The Genetic Cleansing:** If a specific micro-bot loses 3 trades in a row, the Hive-Mind instantly deletes it. At the end of every week, the Hive-Mind uses Genetic Algorithms to "breed" the most profitable micro-bots together to replenish the swarm with 1,000 new, optimized bots for the next week.

### Technical Stack Requested
* **Data & Models:** Python, Ray (for distributed multi-agent processing), DEAP (for Genetic Algorithms), and HuggingFace (for the Observer models).
* **Output:** Please write the blueprint for the Python classes that will orchestrate this Swarm. Start by showing me the `HiveMindController` class, the `MicroBot` base class, and the logic for the `WeeklyGeneticCleansing` function.
