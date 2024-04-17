from transformers import pipeline  # type: ignore



summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# grab links, summarize/sentiment them