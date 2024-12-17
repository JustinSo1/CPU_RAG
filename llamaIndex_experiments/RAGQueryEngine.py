import time

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
# from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer


class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        print("Retrieving documents")
        start_retrieval_time = time.perf_counter()
        nodes = self.retriever.retrieve(query_str)
        end_retrieval_time = time.perf_counter()
        retrieval_time = end_retrieval_time - start_retrieval_time
        print(f"Retrieved documents: {retrieval_time}")

        print("LLM Response Timing")
        start_llm_response_time = time.perf_counter()
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        end_llm_response_time = time.perf_counter()
        llm_response_time = end_llm_response_time - start_llm_response_time
        print(f"LLM Response Timing: {llm_response_time}")

        return response_obj, retrieval_time, llm_response_time
