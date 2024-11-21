import os

from huggingface_hub import login
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.llama_cpp import LlamaCPP

from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document

from utils import read_csv


def get_documents(df):
    documents = [
        Document(
            text=row['passage'],
            # metadata={
            #     'ID': row['ID'],
            #     'Type': row['Type']
            # }
        )
        for _, row in df.iterrows()
    ]
    print(documents)
    return documents


def split_documents(documents):
    splitter = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=0,
        paragraph_separator="\n\n"
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    return nodes


def main():
    model_url = "https://huggingface.co/google/gemma-2-2b-it-GGUF/resolve/main/2b_it_v2.gguf"
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        # model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path="../data/models/gemma2/2b_it_v2.gguf",
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        # generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"repeat_penalty": 1, "penalize_nl": False, "chat_format": "gemma"},
        # transform inputs into Llama2 format
        # messages_to_prompt=messages_to_prompt,
        # completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    # response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
    # print(response.text)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    set_global_tokenizer(
        tokenizer
    )
    embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large")

    text_corpus = read_csv("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet")
    # question_answer = read_csv
    documents = get_documents(text_corpus)
    nodes = split_documents(documents)
    # documents = SimpleDirectoryReader(
    #     "../data/dataset/rag_wikipedia/llamaindex_test"
    # ).load_data()

    # create vector store index
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)

    # set up query engine
    query_engine = index.as_query_engine(llm=llm)

    response = query_engine.query("Who is the 16th president of the USA?")
    print(response)


if __name__ == '__main__':
    main()
