import gradio as gr
import os
import time

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from transformers import pipeline  # type: ignore


load_dotenv(find_dotenv())

summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_pipeline = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# Summ/Sentiment
# radio button links
# then spit out the values
articles = {
    "Native American voices are finally factoring into energy projects – a hydropower ruling is a victory for environmental justice on tribal lands": "https://theconversation.com/native-american-voices-are-finally-factoring-into-energy-projects-a-hydropower-ruling-is-a-victory-for-environmental-justice-on-tribal-lands-224612",
    "Does ‘virtue signaling’ pay off for entrepreneurs? We studied 81,799 Airbnb listings to find out": "https://theconversation.com/does-virtue-signaling-pay-off-for-entrepreneurs-we-studied-81-799-airbnb-listings-to-find-out-226450",
    "Rural students’ access to Wi-Fi is in jeopardy as pandemic-era resources recede": "https://theconversation.com/rural-students-access-to-wi-fi-is-in-jeopardy-as-pandemic-era-resources-recede-225945",
    "Taxes are due even if you object to government policies or doubt the validity of the 16th Amendment’s ratification": "https://theconversation.com/taxes-are-due-even-if-you-object-to-government-policies-or-doubt-the-validity-of-the-16th-amendments-ratification-227208"
}

def summarize(article_title: str, summarize_or_sentiment:str):
    article_link = article[article]
    print(article_link)
    pass


huggingface_demo = gr.Interface(
    fn = summarize,
    inputs = [
        gr.Dropdown(
            articles.keys(), label="articles", info="List of articles to summarize or get sentiment of!"
        ),
        gr.Radio(["summarize", "sentiment"], label="type", info="Summarize or sentiment analysis of article"),
    ],
    outputs = ["text"]
)

huggingface_demo.launch()

# Find API Key

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.

def openai_predict():
    pass

def huggingface_predict():
    pass



def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history):
    response = "**That's cool!**"
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
    )

    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...", show_label=False)

    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    chatbot.like(print_like_dislike, None, None)

demo.queue()
if __name__ == "__main__":
    demo.launch()