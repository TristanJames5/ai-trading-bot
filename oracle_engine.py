import os
import torch
import torch.nn as nn
from transformers import pipeline
import numpy as np
import math
from stable_baselines3 import PPO

# ==========================================
# LAYER 1: THE EYES (Time-Series Transformer)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [batch_size, seq_len, embedding_dim]
        # pe is [max_len, 1, embedding_dim] -> we transpose it to [1, max_len, embedding_dim] or slice appropriately
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return x

class OracleTransformer(nn.Module):
    def __init__(self, feature_dim=13, hidden_dim=64, nhead=4, num_layers=3, num_classes=3):
        super().__init__()
        self.feature_extractor = nn.Linear(feature_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.decoder(x)
        return torch.softmax(x, dim=1)


# ==========================================
# LAYER 2: THE EARS (NLP Sentiment Engine)
# ==========================================
class NewsSentimentEngine:
    def __init__(self):
        try:
            self.nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception as e:
            print(f"Warning: Could not load FinBERT natively ({e})")
            self.nlp = None

    def analyze_headlines(self, headlines):
        if not self.nlp or not headlines:
            return 0.0, "CONFIRM"

        total_score = 0.0
        for headline in headlines:
            result = self.nlp(headline)[0]
            label = result['label']
            score = result['score']
            
            if label == 'positive':
                total_score += score
            elif label == 'negative':
                total_score -= score
                
        avg_score = total_score / len(headlines)
        
        if avg_score < -0.7 or avg_score > 0.7:
            return avg_score, "HALT TRADING"
        else:
            return avg_score, "CONFIRM"


# ==========================================
# LAYER 3: THE BRAIN (PPO Reinforcement Learning)
# ==========================================
class OracleEngine:
    def __init__(self, transformer_path="C:\\Users\\JAYLO\\Downloads\\oracle_phase3.zip", ppo_model_path="sentinel_model.zip"):
        print("Initializing Phase 3 Oracle Architecture...")
        
        self.transformer = OracleTransformer()
        print(f"Loading Transformer Weights from {transformer_path}...")
        try:
            state_dict = torch.load(transformer_path)
            self.transformer.load_state_dict(state_dict)
            print("Transformer loaded successfully!")
        except Exception as e:
            print(f"Failed to load transformer: {e}")
            
        self.transformer.eval()
        self.sentiment_engine = NewsSentimentEngine()
        
        print(f"Loading PPO Model from {ppo_model_path}...")
        try:
            self.ppo_model = PPO.load(ppo_model_path)
            self.ppo_loaded = True
            print("PPO Model Loaded Successfully!")
        except Exception as e:
            print(f"Warning: Could not load PPO model from {ppo_model_path}: {e}")
            self.ppo_loaded = False

    def predict_action(self, recent_candles, recent_news):
        with torch.no_grad():
            x_tensor = torch.tensor(recent_candles, dtype=torch.float32).unsqueeze(0)
            probs = self.transformer(x_tensor)[0].numpy()
            
        bull_prob, bear_prob, side_prob = probs[0], probs[1], probs[2]

        sentiment_score, circuit_breaker = self.sentiment_engine.analyze_headlines(recent_news)

        if circuit_breaker == "HALT TRADING":
            print(f"CIRCUIT BREAKER TRIPPED! Extreme Sentiment: {sentiment_score:.2f}")
            return "CUT", 0

        observation = np.array([bull_prob, bear_prob, side_prob, sentiment_score], dtype=np.float32)
        
        if self.ppo_loaded:
            # We assume the PPO model expects an observation space of size 4 or it will throw an error
            # If the PPO from Phase 2 had a different obs shape, we will mock it for this test.
            try:
                action, _states = self.ppo_model.predict(observation, deterministic=True)
                action = int(action)
            except Exception as e:
                print(f"PPO obs space mismatch ({e}). Mocking decision.")
                action = np.argmax(probs)
        else:
            action = np.argmax(probs)

        leverage_map = {0: 0, 1: 1, 2: 5, 3: 10, 4: 1, 5: 5, 6: 10}
        dir_map = {0: "CUT", 1: "BUY", 2: "BUY", 3: "BUY", 4: "SELL", 5: "SELL", 6: "SELL"}

        if action not in leverage_map:
            if bull_prob > bear_prob:
                action = 1
            else:
                action = 4

        return dir_map[action], leverage_map[action]

if __name__ == "__main__":
    oracle = OracleEngine()
    
    # 13 features for the transformer (instead of 5)
    mock_candles = np.random.rand(100, 13) 
    mock_news = ["Federal reserve cuts interest rates unexpectedly!", "Market surges as inflation drops."]
    
    print("\nEvaluating Market State...")
    direction, leverage = oracle.predict_action(mock_candles, mock_news)
    
    print(f"\n=> ORACLE DECISION: {direction} (Leverage: {leverage}%)")
