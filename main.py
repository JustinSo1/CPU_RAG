import pickle

import scann
import multiprocessing
import statistics
import subprocess

import numpy as np
import pandas as pd
# import transformers
# from transformers import (AutoModelForCausalLM,
#                           AutoTokenizer,
#                           BitsAndBytesConfig,
#                          )
from sentence_transformers import SentenceTransformer

from gemma_wrapper import GemmaCPPPython
from llama_wrapper import LlamaCpp
from utils import map2embeddings, generate_summary_and_answer, read_csv


# import bitsandbytes as bnb

class AIAssistant:
    """An AI assistant that interacts with users by providing answers based on a provided knowledge base"""

    def __init__(self, gemma_model, embeddings_name="thenlper/gte-large", temperature=0.4, role="expert"):
        """Initialize the AI assistant."""
        # Initialize attributes
        self.searcher = None
        self.embeddings = None
        self.embeddings_name = embeddings_name
        self.knowledge_base = []
        self.temperature = temperature
        self.role = role

        # Initialize Gemma model (it can be transformer-based or any other)
        self.gemma_model = gemma_model

        # Load the embedding model
        self.embedding_model = SentenceTransformer(self.embeddings_name)

    def store_knowledge_base(self, knowledge_base):
        """Store the knowledge base"""
        self.knowledge_base = knowledge_base

    def learn_knowledge_base(self, knowledge_base):
        """Store and index the knowledge based to be used by the assistant"""
        # Storing the knowledge base
        self.store_knowledge_base(knowledge_base)

        # Load and index the knowledge base
        print("Indexing and mapping the knowledge base:")
        embeddings = map2embeddings(self.knowledge_base, self.embedding_model)
        self.embeddings = np.array(embeddings).astype(np.float32)

        # Instantiate the searcher for similarity search
        self.index_embeddings()

    def index_embeddings(self):
        """Index the embeddings using ScaNN """
        # print("EMBEDDINGS SHAPE", self.embeddings.shape)
        self.searcher = (
            scann.scann_ops_pybind.builder(db=self.embeddings, num_neighbors=10, distance_measure="dot_product")
            .tree(num_leaves=int(self.embeddings.shape[0] ** 0.5),  # of partitions
                  num_leaves_to_search=int(self.embeddings.shape[0] ** 0.5),
                  min_partition_size=10,
                  # quantize_centroids=True,
                  training_sample_size=self.embeddings.shape[0])
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(100)
            .build()
        )

    def query(self, query):
        """Query the knowledge base of the AI assistant."""
        # Generate and print an answer to the query
        result, answer_time, scann_time, prompt_time = generate_summary_and_answer(query,
                                                                                   self.knowledge_base,
                                                                                   self.searcher,
                                                                                   self.embedding_model,
                                                                                   self.gemma_model,
                                                                                   temperature=self.temperature,
                                                                                   role=self.role)
        #         print(answer)
        return result, answer_time, scann_time, prompt_time

    def set_temperature(self, temperature):
        """Set the temperature (creativity) of the AI assistant."""
        self.temperature = temperature

    def set_role(self, role):
        """Define the answering style of the AI assistant."""
        self.role = role

    def save_embeddings(self, filename="embeddings.npy"):
        """Save the embeddings to disk"""
        np.save(filename, self.embeddings)

    def load_embeddings(self, filename="embeddings.npy"):
        """Load the embeddings from disk and index them"""
        self.embeddings = np.load(filename)
        # Re-instantiate the searcher
        self.index_embeddings()


def run_rag_pipeline(model, n_threads, questions, answers, embeddings_name,
                     tokenizer, compressed_weights, text_corpus):
    # Create an instance of the class AIAssistant based on Gemma C++
    ai_assistant = AIAssistant(
        gemma_model=model, temperature=0.0,
        embeddings_name=embeddings_name
    )
    print(f"Running AI Assistant with {n_threads} threads")
    # Loading the previously prepared knowledge base and embeddings
    knowledge_base = text_corpus.tolist()

    # Map the intended knowledge base to embeddings and index it
    # gemma_ai_assistant.learn_knowledge_base(knowledge_base=knowledge_base)
    # Save the embeddings to disk (for later use)
    # gemma_ai_assistant.save_embeddings()

    # Uploading the knowledge base and embeddings to the AI assistant
    ai_assistant.store_knowledge_base(knowledge_base=knowledge_base)
    ai_assistant.load_embeddings(filename="data/embeddings_bio.npy")
    # # Start the logger running in a background process. It will keep running until you tell it to stop.
    # # We will save the CPU and GPU utilisation stats to a CSV file every 0.2 seconds.
    # !rm -f log_compute.csv
    logger_fname = f'data/log_compute{n_threads}.csv'
    logger_pid = subprocess.Popen(
        ['python', 'log_gpu_cpu_stats.py',
         logger_fname,
         '--loop', '1',  # Interval between measurements, in seconds (optional, default=1)
         ])
    print('Started logging compute utilisation')
    answer_time_arr, scann_time_arr, prompt_time_arr = [], [], []
    result_arr = []
    i = 0
    for i, (question, answer) in enumerate(zip(questions, answers), i):
        print(f"Question #{i}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        result, answer_time, scann_time, prompt_time = ai_assistant.query(question)
        answer_time_arr.append(answer_time)
        scann_time_arr.append(scann_time)
        prompt_time_arr.append(prompt_time)
        result_arr.append(result)
        print("Result:" + result)
        i += 1
        # break
        if i == 50:
            break
        # End the background process logging the CPU and GPU utilisation.
    logger_pid.terminate()
    avg_scann_time = statistics.fmean(scann_time_arr)
    avg_prompt_time = statistics.fmean(prompt_time_arr)
    avg_answer_time = statistics.fmean(answer_time_arr)
    print('Terminated the compute utilisation logger background process')
    print(f"Avg scann time {avg_scann_time}")
    print(f"Avg prompt time {avg_prompt_time}")
    print(f"Avg answer time {avg_answer_time}")
    return avg_scann_time, avg_prompt_time, avg_answer_time, result_arr


def main():
    print(f"Current machine only has {multiprocessing.cpu_count()} cores")
    text_corpus = read_csv("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet")
    question_answer = read_csv("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")

    # text_corpus = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-bioasq/data/passages.parquet/part.0.parquet")
    # question_answer = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-bioasq/data/test.parquet/part.0.parquet")
    questions = question_answer['question'].tolist()
    answers = question_answer['answer'].tolist()
    embeddings_name = "thenlper/gte-large"
    tokenizer = "data/gemma-gemmacpp-2b-it-sfp-v4/tokenizer.spm"
    compressed_weights = "data/gemma-gemmacpp-2b-it-sfp-v4/2b-it-sfp.sbs"
    model = "2b-it"
    # max_threads = 30
    # Loading the previously prepared knowledge base and embeddings
    # knowledge_base = text_corpus['passage'].tolist()
    # print(knowledge_base)
    chunk_size = 100
    df = text_corpus['passage'].apply(lambda line: [line[i:i + chunk_size] for i in range(0, len(line), chunk_size)])
    df = df.explode()
    avg_stats_dict = {}
    # for i in range(1, max_threads + 1):
    avg_scann_time, avg_prompt_time, avg_answer_time, results = run_rag_pipeline(
        model=GemmaCPPPython(tokenizer, compressed_weights),
        # model=LlamaCpp(n_threads=i),
        n_threads=-1,
        questions=questions,
        answers=answers,
        embeddings_name=embeddings_name,
        tokenizer=tokenizer,
        compressed_weights=compressed_weights,
        text_corpus=df)
    avg_stats_dict[f"{16}_threads"] = {
        "avg_scann_time": avg_scann_time,
        "avg_prompt_time": avg_prompt_time,
        "avg_answer_time": avg_answer_time
    }
    with open('gemma_cpp_recom_bio.pickle', 'wb') as handle:
        pickle.dump(avg_stats_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('results_bio.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # break
    avg_stats_df = pd.DataFrame(avg_stats_dict)
    print(avg_stats_df)
    avg_stats_df.to_csv('gemma_cpp_config_bio.csv', index=False)


if __name__ == '__main__':
    main()
