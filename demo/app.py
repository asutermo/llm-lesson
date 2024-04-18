import gradio as gr
import os
import requests
import time

from bs4 import BeautifulSoup  # type: ignore

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from transformers import pipeline  # type: ignore

# Load keys
load_dotenv(find_dotenv())

# preload models
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

articles = {
    "Native American voices are finally factoring into energy projects – a hydropower ruling is a victory for environmental justice on tribal lands": "https://theconversation.com/native-american-voices-are-finally-factoring-into-energy-projects-a-hydropower-ruling-is-a-victory-for-environmental-justice-on-tribal-lands-224612",
    "Does ‘virtue signaling’ pay off for entrepreneurs? We studied 81,799 Airbnb listings to find out": "https://theconversation.com/does-virtue-signaling-pay-off-for-entrepreneurs-we-studied-81-799-airbnb-listings-to-find-out-226450",
    "Rural students’ access to Wi-Fi is in jeopardy as pandemic-era resources recede": "https://theconversation.com/rural-students-access-to-wi-fi-is-in-jeopardy-as-pandemic-era-resources-recede-225945",
    "Taxes are due even if you object to government policies or doubt the validity of the 16th Amendment’s ratification": "https://theconversation.com/taxes-are-due-even-if-you-object-to-government-policies-or-doubt-the-validity-of-the-16th-amendments-ratification-227208"
}

def parse_html(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 403:
        raise Exception("Unable to access content.")
    soup = BeautifulSoup(response.text, "html.parser")
    article_body_html = soup.find("div", itemprop="articleBody")
    if article_body_html:
        return article_body_html.get_text(strip=False)
    else:
        raise Exception("No article body found.")

def summarize(article_title: str, summarize_or_sentiment:str) -> str:
    article_link = articles[article_title]
    article_text = parse_html(article_link)

    if summarize_or_sentiment == "summarize":
        summary = summarizer_pipeline(article_text[:1024], max_length=150, min_length=40)
        return summary[0]['summary_text']
    else:
        sentiment = sentiment_pipeline(article_text[:514])
        return r"Sentiment: " + sentiment[0]['label'] + r" with a confidence of " + str(sentiment[0]['score'])

def huggingface_demo() -> gr.Interface:
    sorted_article_titles = sorted(articles.keys())
    return gr.Interface(
        fn = summarize,
        inputs = [
            gr.Dropdown(
                articles.keys(), label="article_title", info="List of articles to summarize or get sentiment of!"
            ),
            gr.Radio(["summarize", "sentiment"], label="summarize_or_sentiment", info="Summarize or sentiment analysis of article"),
        ],
        outputs = "text",
        examples=[[sorted_article_titles[0], "summarize"], [sorted_article_titles[1], "sentiment"]]
    )

def huggingface_sentiment_demo() -> gr.Interface:
    return gr.Interface.from_pipeline(sentiment_pipeline)


def huggingface_summary_demo() -> gr.Interface:
    return gr.Interface.from_pipeline(summarizer_pipeline)

def huggingface_pipeline_demo() -> gr.Interface:
    pass

default_personality = "You are an instructional, informative, and kind AI assistant."
current_personality = default_personality
openai_client = OpenAI()
openapi_messages = []

def chat(personality: str, content: str) -> str:
    global current_personality, default_personality, openapi_messages

    if not personality: # Set default chat personality
        personality = default_personality
        current_personality = default_personality
        openapi_messages = [
            {"role": "system", "content": personality},
        ]        
    elif personality != current_personality: # reset with new chat personality
        current_personality = personality
        openapi_messages = [
            {"role": "system", "content": personality},
        ]
    if not content:
        return None
    
    openapi_messages.append({"role": "user", "content": content})
    response = openai_client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=openapi_messages
    )
    bot_response = response.choices[0].message.content
    openapi_messages.append({"role": "assistant", "content": bot_response})
    return bot_response

def openapi_demo() -> gr.Interface:
    return gr.Interface(
        chat,
        inputs=[gr.Textbox(lines=2, label="ChatGPT Personality", value="You are an instructional, informative, and kind AI assistant."), gr.Textbox(lines=5, label="ChatGPT 3.5 Turbo")],
        outputs=gr.Textbox(label="Reply"),
        title="Gradio and ChatGPT 3.5 Turbo"
    )

def main_ui() -> gr.TabbedInterface:
    return gr.TabbedInterface(
        [
            huggingface_sentiment_demo(),
            huggingface_summary_demo(),
            huggingface_demo(),
            openapi_demo()
        ],
        [
            "Simple Sentiment",
            "Simple Summarization",
            "Hugging Face Pipelines",
            "OpenAPI Demo"
        ],
        title="Demo"
    )


if __name__ == "__main__":
    main_ui().launch()
    
# Find API Key

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.

def openai_predict():
    pass

# def huggingface_predict():
#     pass



# def print_like_dislike(x: gr.LikeData):
#     print(x.index, x.value, x.liked)

# def add_message(history, message):
#     for x in message["files"]:
#         history.append(((x,), None))
#     if message["text"] is not None:
#         history.append((message["text"], None))
#     return history, gr.MultimodalTextbox(value=None, interactive=False)

# def bot(history):
#     response = "**That's cool!**"
#     history[-1][1] = ""
#     for character in response:
#         history[-1][1] += character
#         time.sleep(0.05)
#         yield history


# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot(
#         [],
#         elem_id="chatbot",
#         bubble_full_width=False,
#     )

#     chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)

#     chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
#     bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
#     bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

#     chatbot.like(print_like_dislike, None, None)

# demo.queue()
# if __name__ == "__main__":
#     demo.launch()