import time
import re

from urllib.parse import urlparse

import feedparser
import gradio as gr
import requests
from bs4 import BeautifulSoup  # type: ignore
from dotenv import find_dotenv, load_dotenv
from langchain.docstore.document import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.utilities import ApifyWrapper
from langchain_community.document_loaders import ApifyDatasetLoader
from langsmith import traceable, wrappers
from openai import OpenAI
from transformers import pipeline  # type: ignore

# Load keys
load_dotenv(find_dotenv())

# preload models
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_pipeline = pipeline(
    "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
)

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
    content = soup.find('article') or \
              soup.find('div', id="maincontent") or \
              soup.find('bsp-story-page') or \
              soup.find('section', class_="text-block")
    if content:
        article_body_html = content.get_text(strip=True)
    else:
        article_body_html = None
  
    if article_body_html:
        print(article_body_html)
        clean = re.compile('<.*?>')
        article_body_html = re.sub(clean, '', article_body_html)
        return article_body_html
    else:
        return 'Unable to parse summary'

FEED_URL = "https://news.ycombinator.com/rss"
feed = feedparser.parse(FEED_URL)
SUPPORTED_URLS = ["apnews.com", "bbc.com", "www.theguardian.com", "www.nature.com"]
articles = {entry.title: entry.link for entry in feed.entries if entry.link if urlparse(entry.link).netloc in SUPPORTED_URLS} 

def summarize(article_title: str, summarize_or_sentiment: str) -> str:
    article_link = articles[article_title]
    print(f'Accessing article at {article_link}')
    article_text = parse_html(article_link)

    if summarize_or_sentiment == "summarize":
        summary = summarizer_pipeline(
            article_text[:1024], max_length=150, min_length=40
        )
        return summary[0]["summary_text"]
    else:
        sentiment = sentiment_pipeline(article_text[:514])
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


def huggingface_sentiment_demo() -> gr.Interface:
    return gr.Interface.from_pipeline(sentiment_pipeline, examples=["I enjoy this!", "I hate this!", "I am neutral."])

default_personality = "You are an instructional, informative, and kind AI assistant."
current_personality = default_personality
openai_client = OpenAI()
openapi_messages = []


def chat(personality: str, content: str) -> str:
    global current_personality, default_personality, openapi_messages

    if not personality:  # Set default chat personality
        personality = default_personality
        current_personality = default_personality
        openapi_messages = [
            {"role": "system", "content": personality},
        ]
    elif personality != current_personality:  # reset with new chat personality
        current_personality = personality
        openapi_messages = [
            {"role": "system", "content": personality},
        ]
    if not content:
        return None

    openapi_messages.append({"role": "user", "content": content})
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=openapi_messages
    )
    bot_response = response.choices[0].message.content
    openapi_messages.append({"role": "assistant", "content": bot_response})
    return bot_response


def chat_demo_2() -> gr.Blocks:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    openapi_chatbot_messages = [
        {
            "role": "system",
            "content": "You are an instructional, informative, and kind AI assistant.",
        }
    ]

    def user(user_message, history):
        openapi_chatbot_messages.append({"role": "user", "content": user_message})
        return "", history + [[user_message, None]]

    def bot(history):
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=openapi_chatbot_messages
        )
        bot_message = response.choices[0].message.content
        openapi_messages.append(
            {"role": "assistant", "content": openapi_chatbot_messages}
        )

        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.05)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

    chat_demo_2.queue()
    return chat_demo_2


def openapi_demo() -> gr.Interface:
    return gr.Interface(
        chat,
        inputs=[
            gr.Textbox(
                lines=2,
                label="ChatGPT Personality",
                value="You are an instructional, informative, and kind AI assistant.",
            ),
            gr.Textbox(lines=5, label="ChatGPT 3.5 Turbo", value="Tell me about yourself!"),
        ],
        outputs=gr.Textbox(label="Reply"),
        title="Gradio and ChatGPT 3.5 Turbo",
    )


@traceable  # Auto-trace this function
def trace_oai_pipeline(user_input: str):
    langsmith_client = wrappers.wrap_openai(OpenAI())
    result = langsmith_client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}], model="gpt-3.5-turbo"
    )
    return result.choices[0].message.content


def langsmith_demo() -> gr.Interface:
    return gr.Interface(
        trace_oai_pipeline,
        inputs=[
            gr.Textbox(
                lines=2,
                label="ChatGPT Trace",
                value="Write an obfuscated hello world app in python.",
            )
        ],
        outputs=gr.Textbox(label="Reply"),
        title="Gradio and ChatGPT 3.5 Turbo with Langsmith Tracing",
    )


apify = ApifyWrapper()


def llm_qa(query: str) -> str:
    # scraper if needed, takes awhile
    # loader = apify.call_actor(
    #     actor_id="apify/website-content-crawler",
    #     run_input={
    #         "startUrls": [{"url": "https://python.langchain.com/docs/use_cases/"}]
    #     },
    #     dataset_mapping_function=lambda item: Document(
    #         page_content=item["text"] or "", metadata={"source": item["url"]}
    #     ),
    # )
    #https://console.apify.com/actors/aYG0l9s7dbB7j3gbS/runs/Kh9YxOBLlMUusL7dO#storage
    loader = ApifyDatasetLoader(
        dataset_id="qm4Ww9p4X7mjggmMC",
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=dataset_item["text"] or "", metadata={"source": dataset_item["url"]}
        ),
    )

    index = VectorstoreIndexCreator().from_loaders([loader])

    # Query the vector store
    result = index.query(query)
    return result


def langchain_demo() -> gr.Interface:
    return gr.Interface(
        llm_qa,
        inputs=[
            gr.Textbox(
                lines=2,
                label="LangChain KB Query",
                value="How do I do synthetic data creation?",
            )
        ],
        outputs=gr.Textbox(label="LangChain KB Reply"),
        title="Gradio and LangChain Knowledge Base Query Answering",
    )


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
