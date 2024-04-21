import gradio as gr
from dotenv import find_dotenv, load_dotenv
from langchain_demo import *
from langsmith_demo import *
from openai_chat import *
from sentiment import *
from summarization_or_sentiment import *

# Load keys
load_dotenv(find_dotenv())

# Preload models
sentiment = get_sentiment_pipeline()
summarizer = get_summarizer_pipeline()


def main_ui() -> gr.TabbedInterface:
    return gr.TabbedInterface(
        [
            huggingface_sentiment_demo(),
            huggingface_demo(),
            openapi_demo(),
            langchain_demo(),
            langsmith_demo(),
        ],
        [
            "Simple Sentiment",
            "Hugging Face Pipelines",
            "OpenAPI Demo",
            "LangChain Demo",
            "LangSmith Demo",
        ],
        title="Demo",
    )


if __name__ == "__main__":
    main_ui().launch()
