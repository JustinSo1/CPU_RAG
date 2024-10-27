import scann
import statistics
import subprocess
import numpy as np
# import transformers
# from transformers import (AutoModelForCausalLM,
#                           AutoTokenizer,
#                           BitsAndBytesConfig,
#                          )
from sentence_transformers import SentenceTransformer

from gemma_wrapper import GemmaCPPPython
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
        self.searcher = (
            scann.scann_ops_pybind.builder(db=self.embeddings, num_neighbors=10, distance_measure="dot_product")
            .tree(num_leaves=min(self.embeddings.shape[0] // 2, 1000),
                  num_leaves_to_search=100,
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


if __name__ == '__main__':
    text_corpus = read_csv("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet")
    question_answer = read_csv("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
    # text_corpus = load_dataset("hf://datasets/rag-datasets/rag-mini-bioasq/data/passages.parquet/part.0.parquet")
    # question_answer = load_dataset("hf://datasets/rag-datasets/rag-mini-bioasq/data/test.parquet/part.0.parquet")
    questions = question_answer['question'].tolist()
    answers = question_answer['answer'].tolist()
    embeddings_name = "thenlper/gte-large"
    tokenizer = "data/gemma-gemmacpp-2b-it-sfp-v4/tokenizer.spm"
    compressed_weights = "data/gemma-gemmacpp-2b-it-sfp-v4/2b-it-sfp.sbs"
    model = "2b-it"
    max_threads = 10

    # for thread_count in range(1, max_threads + 1):
    #     gemma_ai_assistant = AIAssistant(
    #         gemma_model=GemmaCPPPython(tokenizer, compressed_weights, n_threads=thread_count), temperature=0.0,
    #         embeddings_name=embeddings_name
    #     )
    # Loading the previously prepared knowledge base and embeddings
    # knowledge_base = text_corpus['passage'].tolist()

    # Create an instance of the class AIAssistant based on Gemma C++
    gemma_ai_assistant = AIAssistant(
        gemma_model=GemmaCPPPython(tokenizer, compressed_weights, n_threads=1), temperature=0.0,
        embeddings_name=embeddings_name
    )

    # Loading the previously prepared knowledge base and embeddings
    knowledge_base = text_corpus['passage'].tolist()

    # Map the intended knowledge base to embeddings and index it
    # gemma_ai_assistant.learn_knowledge_base(knowledge_base=knowledge_base)

    # Save the embeddings to disk (for later use)
    # gemma_ai_assistant.save_embeddings()

    # Uploading the knowledge base and embeddings to the AI assistant
    gemma_ai_assistant.store_knowledge_base(knowledge_base=knowledge_base)
    gemma_ai_assistant.load_embeddings(filename="data/embeddings.npy")
    # # Start the logger running in a background process. It will keep running until you tell it to stop.
    # # We will save the CPU and GPU utilisation stats to a CSV file every 0.2 seconds.
    # !rm -f log_compute.csv
    logger_fname = 'log_compute.csv'
    logger_pid = subprocess.Popen(
         ['python', 'log_gpu_cpu_stats.py',
          logger_fname,
          '--loop',  '30',  # Interval between measurements, in seconds (optional, default=1)
         ])
    print('Started logging compute utilisation')

    answer_time_arr, scann_time_arr, prompt_time_arr = [], [], []
    i = 0
    for i, (question, answer) in enumerate(zip(questions, answers), i):
        print(f"Question #{i}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        result, answer_time, scann_time, prompt_time = gemma_ai_assistant.query(question)
        answer_time_arr.append(answer_time)
        scann_time_arr.append(scann_time)
        prompt_time_arr.append(prompt_time)
        print("Result:" + result)
        i += 1
        #break
        if i == 500:
            break
    # End the background process logging the CPU and GPU utilisation.
    logger_pid.terminate()
    print('Terminated the compute utilisation logger background process')
    print(f"Avg scann time {statistics.fmean(scann_time_arr)}")
    print(f"Avg prompt time {statistics.fmean(prompt_time_arr)}")
    print(f"Avg answer time {statistics.fmean(answer_time_arr)}")
