import os
import time
from datetime import datetime

import pandas as pd
from llama_index.core import Document, ServiceContext, set_global_service_context, Settings, get_response_synthesizer
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex
from llama_index.core import set_global_tokenizer
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.faiss import FaissVectorStore
from transformers import AutoTokenizer
import faiss
import logging
import sys

from llamaIndex_experiments.RAGQueryEngine import RAGQueryEngine


# from utils import read_parquet


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


def main():
    # os.environ["OPENAI_API_KEY"] = "random"
    # model_url = "https://huggingface.co/google/gemma-2-2b-it-GGUF/resolve/main/2b_it_v2.gguf"
    model_url = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q5_K_M.gguf"
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
        model_kwargs={"repeat_penalty": 1, "penalize_nl": False, "chat_format": "gemma", "n_threads": 10},
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

    text_corpus = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet")
    question_answer = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
    questions = question_answer['question'].tolist()
    answers = question_answer['answer'].tolist()

    Settings.llm = llm
    Settings.embed_model = embed_model
    # Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    # Settings.num_output = 512
    # Settings.context_window = 3900

    llama_index_embeddings_path = "../data/dataset/rag_wikipedia/embeddings/llama_index_embeddings/"

    # create vector store index
    faiss_index = faiss.IndexFlatL2(1024)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    # index_start_build_time = time.perf_counter()
    # index = construct_index(text_corpus, embed_model, llama_index_embeddings_path, vector_store)
    # index_end_build_time = time.perf_counter()
    # logging.log(logging.INFO, f"Index build time: {index_end_build_time - index_start_build_time}s")

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=llama_index_embeddings_path,
                                                   vector_store=vector_store.from_persist_dir(
                                                       persist_dir=llama_index_embeddings_path
                                                   ))
    # # load index
    index = load_index_from_storage(storage_context=storage_context)

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    # query_engine = RetrieverQueryEngine(
    #     retriever=retriever,
    #     response_synthesizer=response_synthesizer,
    #     # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.0)],
    # )
    query_engine = RAGQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    answer_dict = {}
    for i, (question, answer) in enumerate(zip(questions, answers), 0):
        print(f"Question #{i}")
        print(f"Question: {question}")
        print(f"Ground Truth: {answer}")
        response, retrieval_time, llm_response_time = query_engine.custom_query(question)
        print(f"Result: {response}")
        answer_dict[f"Q{i}"] = {"Answer": response,
                                "retrieval_time": retrieval_time,
                                "llm_response_time": llm_response_time
                                }
        if i == 50:
            break
        break
    df = pd.DataFrame.from_dict(answer_dict)
    print(df)
    df.to_csv("llama_index_wiki_gemma-2-2b-it-Q5_K_M.csv", index=False)


if __name__ == '__main__':
    log_file = f"logs/gemma_llamaindex_wiki_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.log"
    logging.basicConfig(level=logging.INFO, filename=log_file,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stderr))
    with open(log_file, "a") as log:
        sys.stderr = log
        main()
