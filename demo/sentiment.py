import gradio as gr
from transformers import TextClassificationPipeline, pipeline  # type: ignore

__all__ = ["get_sentiment_pipeline", "huggingface_sentiment_demo"]


def get_sentiment_pipeline() -> TextClassificationPipeline:
    """Create a text classification - sentiment - pipeline"""
    return pipeline(
        "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
    )


def huggingface_sentiment_demo() -> gr.Interface:
    """Use gradio + huggingface intergration"""
    return gr.Interface.from_pipeline(
        get_sentiment_pipeline(),
        examples=["I enjoy this!", "I hate this!", "I am neutral."],
    )


if __name__ == "__main__":
    huggingface_sentiment_demo().launch()
