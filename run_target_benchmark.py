# from target_benchmark.evaluators import TARGET, get_task_names
# from target_benchmark.retrievers import AbsStandardEmbeddingRetriever
# from target_benchmark.tasks import QuestionAnsweringTask

from CustomRetriever import Retriever
from CustomGenerator import CustomGenerator

import pandas as pd
from datasets import load_dataset

from target.target_benchmark.evaluators.TARGET import TARGET
from target.target_benchmark.tasks.QuestionAnsweringTask import QuestionAnsweringTask

if __name__ == '__main__':
    splits = {'train': 'fetaQA-v1_train.jsonl',
              'validation': 'fetaQA-v1_dev.jsonl'}
    # validation split
    validation_dataset = load_dataset("DongfuJiang/FeTaQA", split="validation")

    # Initialize the Retriever
    retriever = Retriever()
    # Initialize the Generator
    generator = CustomGenerator()

    # Step 2: Test the Retriever
    query = "Who won the gold medal in Serbia?"
    dataset_name = "fetaqa_dataset"

    # Generate embedding for a query
    query_embedding = retriever.embed_query(query=query, dataset_name=dataset_name)
    print(f"Query Embedding: {query_embedding}")

    corpus_entry = {
        "database_id": ["0"],
        "table_id": ["Serbia_at_the_European_Athletics_Championships_2"],
        "table": [["Name", "Country"], ["Alice", "USA"], ["Bob", "Serbia"]],
        "context": [{"section_title": "Indoor -- List of Medalists"}],
    }

    # Generate embedding for a corpus entry
    corpus_embedding = retriever.embed_corpus(dataset_name=dataset_name, corpus_entry=corpus_entry)
    print(f"Corpus Embedding: {corpus_embedding}")

    # Initialize the evaluation task
    qa_task = QuestionAnsweringTask(task_generator=generator)
    target_fetaqa = TARGET(downstream_tasks=qa_task)

    # Run evaluation
    performance = target_fetaqa.run(retriever=retriever, split="validation", top_k=10)
    print(f"Performance: {performance}")
