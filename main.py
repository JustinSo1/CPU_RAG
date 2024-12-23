import logging
import sys
from datetime import datetime
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
from utils import create_llama_cpp_model


def main():
    # os.environ["OPENAI_API_KEY"] = "random"
    # model_url = "https://huggingface.co/google/gemma-2-2b-it-GGUF/resolve/main/2b_it_v2.gguf"
    model_url = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q5_K_M.gguf"
    llm = create_llama_cpp_model(model_url)
    load_dotenv('.env')

    # # Define llm parameters
    #    llm = AzureChatOpenAI(
    #        deployment_name=os.environ['MODEL'],
    #        openai_api_version=os.environ['API_VERSION'],
    #        openai_api_key=os.environ['OPENAI_API_KEY'],
    #        azure_endpoint=os.environ['OPENAI_API_BASE'],
    #        openai_organization=os.environ['OPENAI_ORGANIZATION']
    #    )
    #    llm_predictor = LangChainLLM(llm=llm)

    # response = llm_predictor.complete("Hello! Can you tell me a poem about cats and dogs?")
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

    token_counter = TokenCountingHandler(
        tokenizer=tokenizer
    )

    #    Settings.llm = llm_predictor
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.callback_manager = CallbackManager([token_counter])
    # Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    # Settings.num_output = 512
    # Settings.context_window = 3900

    llama_index_embeddings_path = "data/dataset/rag_wikipedia/embeddings/llama_index_embeddings/"

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
    logging.log(logging.INFO, f"""
        Embedding Tokens: {token_counter.total_embedding_token_count}
        LLM Prompt Tokens: {token_counter.prompt_llm_token_count}
        LLM Completion Tokens: {token_counter.completion_llm_token_count}
        Total LLM Token Count: {token_counter.total_llm_token_count}
        """
                )
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
    # df.to_csv("llama_index_wiki_gemma-2-2b-it-Q5_K_M.csv")


if __name__ == '__main__':
    log_file = f"logs/gemma_llamaindex_wiki_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.log"
    logging.basicConfig(level=logging.INFO, filename=log_file,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stderr))
    with open(log_file, "a") as log:
        sys.stderr = log
        main()
