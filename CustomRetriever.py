from target_benchmark.retrievers import AbsStandardEmbeddingRetriever
from sentence_transformers import SentenceTransformer
from typing import Dict
import numpy as np


class Retriever(AbsStandardEmbeddingRetriever):
    def __init__(self, expected_corpus_format: str = "nested array", model_name: str = "thenlper/gte-large", **kwargs):
        super().__init__(expected_corpus_format=expected_corpus_format)
        self.embedding_model = SentenceTransformer(model_name)

    # return the embeddings for the query as a numpy array
    def embed_query(self, query: str, dataset_name: str, **kwargs) -> np.ndarray:
        return self.embedding_model.encode(query, convert_to_numpy=True)

    # returns embedding of the passed in table as a numpy array
    def embed_corpus(self, dataset_name: str, corpus_entry: Dict) -> np.ndarray:
        # Extract context and table information
        context = corpus_entry.get("context", [{}])
        table = corpus_entry.get("table", None)

        # Extract section title or other context metadata
        print(context)
        if not isinstance(context, list):
            context = [context]
        section_title = context[0].get("section_title", "") if context else ""

        # Preprocess the table into text (simplified example)
        if isinstance(table, list):  # Assuming table is a list of rows
            table_text = "\n".join(["\t".join(map(str, row)) for row in table])
        else:
            table_text = str(table)  # Fallback in case the format is unexpected

        # Combine context and table content into a single string
        combined_text = f"Section: {section_title}\nTable Content:\n{table_text}"

        # Generate embedding for the combined text
        return self.embedding_model.encode(combined_text, convert_to_numpy=True)
