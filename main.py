from fastapi import FastAPI, Request, HTTPException
import uvicorn
from pydantic import BaseModel
import logging
from typing import Optional

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Trading Bot Webhook Receiver")

# Define the expected payload from TradingView
class TradingViewAlert(BaseModel):
    passphrase: str
    asset: str
    mode: str  # Scalp, Day Trade, Swing, Positional
    price: float
    ema_trend: str
    zone: str  # Premium, Discount, Equilibrium
    fvg_detected: bool = False
    ob_detected: bool = False
    request_bias: bool = False

WEBHOOK_PASSPHRASE = "your_secure_passphrase_here"

@app.get("/")
def read_root():
    return {"status": "AI Trading Bot is running"}

@app.post("/webhook")
async def receive_webhook(alert: TradingViewAlert):
    if alert.passphrase != WEBHOOK_PASSPHRASE:
        logger.warning(f"Unauthorized webhook attempt: {alert.passphrase}")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    logger.info(f"--- New Alert: {alert.asset} ({alert.mode}) ---")
    logger.info(f"Price: {alert.price} | EMA Trend: {alert.ema_trend} | Zone: {alert.zone}")
    logger.info(f"FVG: {alert.fvg_detected} | OB: {alert.ob_detected}")
    
    # Example News Filter Logic (Placeholder)
    # Here you would call an API like ForexFactory or News API to check for high-impact news
    # if check_for_high_impact_news():
    #     logger.warning("High impact news detected. Trading paused.")
    #     return {"status": "skipped", "reason": "high_impact_news"}

    response_data = {"status": "success", "message": "Alert received"}

    if alert.request_bias:
        # AI BIAS LOGIC (To be replaced with ML model prediction)
        ai_bias = "Neutral"
        if alert.zone == "Discount" and alert.ema_trend == "Bullish" and alert.ob_detected:
            ai_bias = "Strong Bullish"
        elif alert.zone == "Premium" and alert.ema_trend == "Bearish" and alert.ob_detected:
            ai_bias = "Strong Bearish"
            
        logger.info(f"AI Evaluation -> Bias: {ai_bias}")
        response_data["ai_bias"] = ai_bias

    return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
