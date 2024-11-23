# you can run `get_task_names()` to get all available tasks
import os
from typing import List, Tuple, Iterable, Dict

from target.target_benchmark.evaluators.TARGET import TARGET
from target.target_benchmark.generators.AbsGenerator import AbsGenerator
from target.target_benchmark.retrievers import AbsCustomEmbeddingRetriever
from target.target_benchmark.retrievers.llama_index.LlamaIndexRetriever import LlamaIndexRetriever
from target.target_benchmark.tasks.QuestionAnsweringTask import QuestionAnsweringTask


class YourCustomGenerator(AbsGenerator):
    # returns the answer to the query
    def generate(self, table_str: str, query: str) -> str:
        return ""


class YourRetriever(AbsCustomEmbeddingRetriever):
    # you can specify a `expected_corpus_format`
    # (ie nested array, dictionary, dataframe, etc.),
    # the corpus tables will be converted to this format
    # before passed into the `embed_corpus` function.
    def __init__(self, expected_corpus_format: str = "nested array", **kwargs):
        super().__init__(expected_corpus_format=expected_corpus_format)

    # returns a list of tuples, each being (database_id, table_id) of the retrieved table
    def retrieve(self, query: str, dataset_name: str, top_k: int) -> List[Tuple]:
        return [(1, 1)]

    # returns nothing since the embedding persistence is dealt with within this function.
    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]) -> None:
        return


def main():
    qa_task = QuestionAnsweringTask(task_generator=YourCustomGenerator())
    target_evaluator = TARGET(downstream_tasks=qa_task)
    # specify a task and a dataset to run evaluations on.
    # target_fetaqa = TARGET(("Table Retrieval Task", "fetaqa"))
    # create a new retriever object
    llamaindex_retriever = YourRetriever()
    # run the evaluation!
    performance = target_evaluator.run(retriever=llamaindex_retriever, split="train", top_k=10)
    print(performance)
    #
    # # specify a task and a dataset to run evaluations on.
    # target_fetaqa = TARGET(("Table Retrieval Task", "fetaqa"))
    # # create a new retriever object
    # llamaindex_retriever = LlamaIndexRetriever()
    # # run the evaluation!
    # performance = target_fetaqa.run(retriever=llamaindex_retriever, split="test", top_k=10)
    # print(performance)
    #
    # # if you'd like, you can also persist the retrieval and downstream generation results
    # performance = target_fetaqa.run(retriever=llamaindex_retriever, split="test", top_k=10,
    #                                 retrieval_results_file="./retrieval.jsonl",
    #                                 downstream_results_file="./downstream.jsonl")


if __name__ == '__main__':
    main()
