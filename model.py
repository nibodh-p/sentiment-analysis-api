# %%
import xml.etree.ElementTree as ET
import pandas as pd
import requests


def fetch_live_headlines(topic: str = "financial markets", limit: int = 20) -> pd.DataFrame:
    """Fetches real-time headlines from Google News RSS for any keyword/topic."""
    formatted_topic = topic.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={formatted_topic}&hl=en-US&gl=US&ceid=US:en"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(rss_url, headers=headers, timeout=10)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch RSS feed: Status {response.status_code}")

    root = ET.fromstring(response.content)
    articles = []

    for item in root.findall(".//item")[:limit]:
        title = item.find("title").text if item.find("title") is not None else ""
        pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
        link = item.find("link").text if item.find("link") is not None else ""

        articles.append({"headline": title, "published_at": pub_date, "source_url": link})

    df = pd.DataFrame(articles)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df


# Quick sanity check inside notebook
test_df = fetch_live_headlines("Artificial Intelligence", limit=5)
test_df[["headline", "published_at"]]

# %%
import torch
from transformers import pipeline


class SentimentPipeline:

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Initializes PyTorch Hugging Face pipeline."""
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Loading transformer model '{model_name}' on device {self.device}...")

        self.classifier = pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
        )

    def predict_dataframe(self, df: pd.DataFrame, text_column: str = "headline") -> pd.DataFrame:
        """Processes an entire DataFrame column through the Transformer model."""
        if df.empty or text_column not in df.columns:
            return df

        texts = df[text_column].tolist()
        results = self.classifier(texts)

        # Standardize labels (positive, neutral, negative)
        sentiments = [res["label"].lower() for res in results]
        confidences = [round(res["score"], 4) for res in results]

        df_out = df.copy()
        df_out["sentiment"] = sentiments
        df_out["confidence_score"] = confidences
        return df_out


# Initialize engine
nlp_engine = SentimentPipeline()

# %%
# Fetch live market news
query_topic = "Tech Stocks"
news_df = fetch_live_headlines(topic=query_topic, limit=20)

#Run Transformer inference
analyzed_df = nlp_engine.predict_dataframe(news_df, text_column="headline")

# Display top 10 results directly in the notebook output
analyzed_df[["headline", "sentiment", "confidence_score"]].head(10)

# %%
import plotly.express as px
import plotly.io as pio

# Switch to notebook_connected renderer
pio.renderers.default = "notebook_connected"

# Prepare data
sentiment_counts = analyzed_df["sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]

# Build Donut Chart
fig_pie = px.pie(
    sentiment_counts,
    values="Count",
    names="Sentiment",
    title=f"Market Sentiment Distribution: '{query_topic}'",
    hole=0.4,
    color="Sentiment",
    color_discrete_map={
        "positive": "#00CC96",
        "neutral": "#636EFA",
        "negative": "#EF553B",
    },
)

fig_pie.show()

# %%
# Filter and sort for the Most Positive headlines
most_positive = analyzed_df[analyzed_df["sentiment"] == "positive"]
most_positive = most_positive.sort_values(by="confidence_score", ascending=False).head(3)

# Filter and sort for the Most Negative headlines
most_negative = analyzed_df[analyzed_df["sentiment"] == "negative"]
most_negative = most_negative.sort_values(by="confidence_score", ascending=False).head(3)

# Display the results clearly in the notebook
print("🟢 TOP 3 POSITIVE CATALYSTS:")
print(most_positive[["headline", "confidence_score"]])

print("\n🔴 TOP 3 NEGATIVE CATALYSTS:")
print(most_negative[["headline", "confidence_score"]])

# %%
# Generate a clean filename based on your topic
filename = f"sentiment_{query_topic.replace(' ', '_').lower()}.csv"

# Export to CSV (index=False prevents Pandas from adding extra row numbers)
analyzed_df.to_csv(filename, index=False)

print(f"✅ Successfully exported {len(analyzed_df)} rows to '{filename}'!")


