import os

import pandas as pd
from datasets import load_dataset
from langchain_openai import AzureChatOpenAI
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.llama_cpp import LlamaCPP
from enum import Enum


class LLMName(Enum):
    GEMMA2_2B_IT = 1
    GPT_4o = 2


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
    # print(documents)
    return documents


def split_documents(documents, embed_model):
    splitter = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=0,
        paragraph_separator="\n\n"
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    # splitter = SemanticSplitterNodeParser.from_defaults(embed_model=embed_model)
    # nodes = splitter.build_semantic_nodes_from_documents(documents, show_progress=True)
    return nodes


def construct_index(text_corpus, embed_model, llama_index_embeddings_path, vector_store):
    print("Building index")
    documents = get_documents(text_corpus)
    nodes = split_documents(documents, embed_model)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True, storage_context=storage_context)

    index.storage_context.persist(persist_dir=llama_index_embeddings_path)

    print("Built index")
    return index


def create_llama_cpp_model(model_url):
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        # model_path="../data/models/gemma2/gemma-2-2b-it.Q5_K_M.gguf",
        temperature=0.1,
        max_new_tokens=256,
        context_window=8192,
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
    return llm


def load_the_dataset(hf_path, ):
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    return ds


def create_question_answer_df(file_name):
    df = pd.read_csv(file_name, index_col=0)
    df = df.transpose()
    df = df['Answer']
    return df


def create_gpt4o_model():
    llm_model = AzureChatOpenAI(
        deployment_name=os.environ['MODEL'],
        openai_api_version=os.environ['API_VERSION'],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        azure_endpoint=os.environ['OPENAI_API_BASE'],
        openai_organization=os.environ['OPENAI_ORGANIZATION']
    )
    llm_predictor = LangChainLLM(llm=llm_model)
    return llm_predictor


def create_llm_model(model_name):
    llm_models = {
        LLMName.GEMMA2_2B_IT: lambda: create_llama_cpp_model(
            "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q5_K_M.gguf"),
        LLMName.GPT_4o: lambda: create_gpt4o_model(),
    }
    return llm_models.get(model_name, lambda: "Invalid arg")()
