import gradio as gr
from dotenv import find_dotenv, load_dotenv
from langchain.docstore.document import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_community.utilities import ApifyWrapper

# Load keys
load_dotenv(find_dotenv())

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
    # https://console.apify.com/actors/aYG0l9s7dbB7j3gbS/runs/Kh9YxOBLlMUusL7dO#storage
    loader = ApifyDatasetLoader(
        dataset_id="qm4Ww9p4X7mjggmMC",
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=dataset_item["text"] or "",
            metadata={"source": dataset_item["url"]},
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


if __name__ == "__main__":
    langchain_demo().launch()
