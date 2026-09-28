# Phase 4: The Apex Evolution Arena (Genetic Algorithm)

*Save this prompt. When you are ready to build the ultimate Phase 4 AI, copy everything below this line and paste it into a new AI chat.*

---

**PROMPT BEGINS HERE:**

Act as an elite Machine Learning Quantitative Developer specializing in Genetic Algorithms (NEAT) and Reinforcement Learning for algorithmic trading. 

I want to build a "Survival of the Fittest" Trading Arena in Python using DEAP (Distributed Evolutionary Algorithms in Python) or a custom Genetic Algorithm framework.

**The Concept: Champion Seeding (Elitism)**
We are going to evolve a new trading AI, but we are not starting from scratch. We are going to seed Generation 1 with my existing, highly profitable "Champion" bots, and force them to compete against random, mutated bots in a historical market simulation.

**Here are the core requirements for the simulation:**

### 1. The Gene Pool (Generation 1)
The simulation will spawn 100 bots in a simulated environment (e.g., 2020-2024 EURUSD and BTC historical data).
* **Bot 1 (The Anchor):** Hard-coded with strict logic (my v4.1 + v5 standard bot).
* **Bot 2 (The Assassin):** My existing PPO-trained Apex Sentinel bot.
* **Bots 3 to 100:** Completely randomized neural networks with random risk parameters, random indicator triggers, and random stop-loss rules.

### 2. The Fitness Function (The Arena Rules)
Each generation will trade the historical data for 'X' simulated months.
The "Fitness Score" (how we decide who lives and dies) must heavily penalize drawdowns and reward consistent risk-adjusted returns (Sharpe/Sortino Ratio), not just raw profit. 
* Any bot that blows its account (drawdown > 20%) is instantly killed.
* The bottom 90% of surviving bots are deleted.

### 3. Crossover & Mutation (Super Soldier Breeding)
* Take the top 10% of surviving bots (which will likely be my Champions + 1 or 2 lucky mutant bots).
* **Crossover:** Combine their weights/genes to create 100 new offspring. The offspring should inherit the risk management of the Champions, but acquire the hidden Alpha discovered by the mutant survivors.
* **Mutation:** Introduce a 1% to 5% random mutation rate in the offspring's neural network to encourage the discovery of new strategies.

### 4. Technical Stack Requested
* **Data:** Pandas / NumPy for rapid vector backtesting (no slow loop iteration).
* **Algorithm:** DEAP library (or PyGAD) for the genetic evolution.
* **Output:** The script should output a saved neural network / weights file of the ultimate "Generation 1000" survivor.

Please write the Python framework and class structures to initialize this Arena. Start with the `ArenaEnv` (the backtester) and the `GeneticBreeder` (the crossover/mutation engine). Explain how we will inject my existing PPO/XGBoost logic into the genes of the starting Champions.
