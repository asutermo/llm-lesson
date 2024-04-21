import re

import feedparser
import gradio as gr
import requests
from bs4 import BeautifulSoup
from sentiment import *
from transformers import SummarizationPipeline, pipeline  # type: ignore

__all__ = ["get_summarizer_pipeline", "huggingface_demo"]


def get_summarizer_pipeline() -> SummarizationPipeline:
    return pipeline("summarization", model="facebook/bart-large-cnn")


# Hacky to support multiple sites
def parse_html(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 403:
        raise Exception("Unable to access content.")
    soup = BeautifulSoup(response.text, "html.parser")

    # div - maincontent - guardian
    # article - nature
    # bsp-story-page - apnews
    # bbc - section - text-block
    content = (
        soup.find("article")
        or soup.find("div", id="maincontent")
        or soup.find("bsp-story-page")
        or soup.find("section", class_="text-block")
    )
    if content:
        article_body_html = content.get_text(strip=True)
    else:
        article_body_html = None

    if article_body_html:
        print(article_body_html)
        clean = re.compile("<.*?>")
        article_body_html = re.sub(clean, "", article_body_html)
        return article_body_html
    else:
        return "Unable to parse summary"


FEED_URL = "https://www.theguardian.com/us/rss"
feed = feedparser.parse(FEED_URL)
articles = {entry.title: entry.link for entry in feed.entries}


def summarize(article_title: str, summarize_or_sentiment: str) -> str:
    article_link = articles[article_title]
    print(f"Accessing article at {article_link}")
    article_text = parse_html(article_link)

    if summarize_or_sentiment == "summarize":
        summary = get_summarizer_pipeline()(
            article_text[:1024], max_length=150, min_length=40
        )
        return summary[0]["summary_text"]
    else:
        sentiment = get_sentiment_pipeline()(article_text[:514])
        return (
            r"Sentiment: "
            + sentiment[0]["label"]
            + r" with a confidence of "
            + str(sentiment[0]["score"])
        )


def huggingface_demo() -> gr.Interface:
    sorted_article_titles = sorted(articles.keys())

    return gr.Interface(
        fn=summarize,
        inputs=[
            gr.Dropdown(
                articles.keys(),
                label="article_title",
                info="List of articles to summarize or get sentiment of!",
            ),
            gr.Radio(
                ["summarize", "sentiment"],
                label="summarize_or_sentiment",
                info="Summarize or sentiment analysis of article",
            ),
        ],
        outputs="text",
        examples=[
            [sorted_article_titles[0], "summarize"],
            [sorted_article_titles[1], "sentiment"],
        ],
    )


if __name__ == "__main__":
    huggingface_demo().launch()
