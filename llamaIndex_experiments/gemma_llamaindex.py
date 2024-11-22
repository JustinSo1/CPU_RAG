import os

import pandas as pd
from llama_index.core import Document, ServiceContext, set_global_service_context, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex
from llama_index.core import set_global_tokenizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.faiss import FaissVectorStore
from transformers import AutoTokenizer
import faiss
import logging
import sys

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
    # print(documents)
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
    # os.environ["OPENAI_API_KEY"] = "random"
    model_url = "https://huggingface.co/google/gemma-2-2b-it-GGUF/resolve/main/2b_it_v2.gguf"
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        # model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path="../data/models/gemma2/2b_it_v2.gguf",
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
    # response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
    # print(response.text)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    set_global_tokenizer(
        tokenizer
    )
    embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large")

    text_corpus = read_csv("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet")
    question_answer = read_csv("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
    questions = question_answer['question'].tolist()
    answers = question_answer['answer'].tolist()

    # question_answer = read_csv
    documents = get_documents(text_corpus)
    nodes = split_documents(documents)
    # documents = SimpleDirectoryReader(
    #     "../data/dataset/rag_wikipedia/llamaindex_test"
    # ).load_data()

    Settings.llm = llm
    Settings.embed_model = embed_model
    # Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    # Settings.num_output = 512
    # Settings.context_window = 3900

    # create vector store index
    # index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
    print(embed_model)
    faiss_index = faiss.IndexFlatL2(1024)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True, storage_context=storage_context)

    index.storage_context.persist(persist_dir="../data/dataset/rag_wikipedia/embeddings/")
    # rebuild storage context
    # storage_context = StorageContext.from_defaults(persist_dir="../data/dataset/rag_wikipedia/embeddings/")
    # load index
    # index = load_index_from_storage(storage_context=storage_context)

    # set up query engine
    query_engine = index.as_query_engine(llm=llm)
    answer_dict = {}
    for i, (question, answer) in enumerate(zip(questions, answers), 0):
        print(f"Question #{i}")
        print(f"Question: {question}")
        print(f"Ground Truth: {answer}")
        response = query_engine.query(question)
        print(f"Result: {response}")
        answer_dict[f"Q{i}"] = {"Answer": response}
        if i == 50:
            break
    df = pd.DataFrame.from_dict(answer_dict)
    print(df)
    df.to_csv("vamos.csv")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename="gemma_llamaindex.log",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    main()
