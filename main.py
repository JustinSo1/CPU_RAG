# Uses RAG pipeline to answer questions
import logging
import sys
import time
from datetime import datetime

from datasets import load_dataset
# Transformers package import needs to be at top or else SIG errors
from transformers import AutoTokenizer
import faiss
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import set_global_tokenizer
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from RAGQueryEngine import RAGQueryEngine
from utils import LLMName, construct_index, create_llm_model


def main(llm_model, experiment_output_file, is_index_created=False):
    # Define llm parameters

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    set_global_tokenizer(
        tokenizer
    )
    embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large")

    question_answer_ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    questions = question_answer_ds['test']['question']
    answers = question_answer_ds['test']['answer']

    token_counter = TokenCountingHandler(
        tokenizer=tokenizer
    )

    Settings.llm = llm_model
    Settings.embed_model = embed_model
    Settings.callback_manager = CallbackManager([token_counter])
    llama_index_embeddings_path = "data/dataset/rag_wikipedia/embeddings/llama_index_embeddings/"

    # create vector store index
    # 1024 is the embedding dimension length
    faiss_index = faiss.IndexFlatL2(1024)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    if not is_index_created:
        text_corpus_ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
        index_start_build_time = time.perf_counter()
        index = construct_index(text_corpus_ds['passages'], embed_model, llama_index_embeddings_path, vector_store)
        index_end_build_time = time.perf_counter()
        logging.log(logging.INFO, f"Index build time: {index_end_build_time - index_start_build_time}s")
    else:
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
                                "llm_response_time": llm_response_time,
                                "embedding_tokens": token_counter.total_embedding_token_count,
                                "llm_prompt_tokens": token_counter.prompt_llm_token_count,
                                "llm_completion_tokens": token_counter.completion_llm_token_count,
                                "total_llm_token_count": token_counter.total_llm_token_count
                                }
        logging.log(logging.INFO, f"""
            Embedding Tokens: {token_counter.total_embedding_token_count}
            LLM Prompt Tokens: {token_counter.prompt_llm_token_count}
            LLM Completion Tokens: {token_counter.completion_llm_token_count}
            Total LLM Token Count: {token_counter.total_llm_token_count}
            """
                    )
        token_counter.reset_counts()
        #        if i == 50:
        #            break
        break
    df = pd.DataFrame.from_dict(answer_dict)
    print(df)
    df.to_csv(experiment_output_file)


if __name__ == '__main__':
    load_dotenv('.env')
    experiment_name = "gemma2-2b-it_rag_wiki"
    llm_model_name = LLMName.GEMMA2_2B_IT
    model_url = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q5_K_M.gguf"

    log_file = f"logs/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log"
    logging.basicConfig(level=logging.INFO, filename=log_file,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    with open(log_file, "a") as log:
        sys.stderr = log
        llm = create_llm_model(llm_model_name)
        main(llm, "data/dataset/rag_wikipedia/results/test.csv", True)
