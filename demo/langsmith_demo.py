import gradio as gr
from dotenv import find_dotenv, load_dotenv
from langsmith import traceable, wrappers
from openai import OpenAI

__all__ = ["langsmith_demo"]

# Load keys
load_dotenv(find_dotenv())


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


if __name__ == "__main__":
    langsmith_demo().launch()
