# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import fetch_live_headlines, SentimentPipeline

app = FastAPI(
    title="Live Market Sentiment API",
    description="Fetches live news headlines and evaluates transformer-based market sentiment.",
    version="1.0.0"
)

# Initialize the NLP transformer engine on server startup
nlp_engine = SentimentPipeline()

class TopicRequest(BaseModel):
    topic: str = "Tech Stocks"
    limit: int = 10

@app.get("/")
def home():
    return {"status": "Online", "message": "Sentiment API is ready."}

@app.post("/analyze-market")
def analyze_market_sentiment(payload: TopicRequest):
    try:
        news_df = fetch_live_headlines(topic=payload.topic, limit=payload.limit)
        if news_df.empty:
            return {"topic": payload.topic, "count": 0, "results": []}
        
        analyzed_df = nlp_engine.predict_dataframe(news_df, text_column="headline")
        analyzed_df["published_at"] = analyzed_df["published_at"].astype(str)
        
        records = analyzed_df.to_dict(orient="records")
        return {
            "topic": payload.topic,
            "count": len(records),
            "results": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    #uvicorn main:app --reload
    