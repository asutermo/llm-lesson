import time

import gradio as gr
from openai import OpenAI

__all__ = ["openapi_demo"]

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
            gr.Textbox(
                lines=5, label="ChatGPT 3.5 Turbo", value="Tell me about yourself!"
            ),
        ],
        outputs=gr.Textbox(label="Reply"),
        title="Gradio and ChatGPT 3.5 Turbo",
    )


if __name__ == "__main__":
    openapi_demo().launch()
