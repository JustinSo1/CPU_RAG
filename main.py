from datasets import load_dataset
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
# import scann
import time
import torch
import statistics

# import transformers
# from transformers import (AutoModelForCausalLM,
#                           AutoTokenizer,
#                           BitsAndBytesConfig,
#                          )
from sentence_transformers import SentenceTransformer

from gemma_wrapper import GemmaCPPPython


# import bitsandbytes as bnb

def define_device():
    """Define the device to be used by PyTorch"""

    # Get the PyTorch version
    torch_version = torch.__version__

    # Print the PyTorch version
    print(f"PyTorch version: {torch_version}", end=" -- ")

    # Check if MPS (Multi-Process Service) device is available on MacOS
    if torch.backends.mps.is_available():
        # If MPS is available, print a message indicating its usage
        print("using MPS device on MacOS")
        # Define the device as MPS
        defined_device = torch.device("mps")
    else:
        # If MPS is not available, determine the device based on GPU availability
        defined_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Print a message indicating the selected device
        print(f"using {defined_device}")

    # Return the defined device
    return defined_device


def get_embedding(text, embedding_model):
    """Get embeddings for a given text using the provided embedding model"""

    # Encode the text to obtain embeddings using the provided embedding model
    embedding = embedding_model.encode(text, show_progress_bar=False)

    # Convert the embeddings to a list of floats and return
    return embedding.tolist()


def map2embeddings(data, embedding_model):
    """Map a list of texts to their embeddings using the provided embedding model"""

    # Initialize an empty list to store embeddings
    embeddings = []

    # Iterate over each text in the input data list
    no_texts = len(data)
    print(f"Mapping {no_texts} pieces of information")
    for i in tqdm(range(no_texts)):
        # Get embeddings for the current text using the provided embedding model
        embeddings.append(get_embedding(data[i], embedding_model))

    # Return the list of embeddings
    return embeddings


def clean_text(txt, EOS_TOKEN):
    """Clean text by removing specific tokens and redundant spaces"""
    txt = (txt
           .replace(EOS_TOKEN, "")  # Replace the end-of-sentence token with an empty string
           .replace("**", "")  # Replace double asterisks with an empty string
           .replace("<pad>", "")  # Replace "<pad>" with an empty string
           .replace("  ", " ")  # Replace double spaces with single spaces
           ).strip()  # Strip leading and trailing spaces from the text
    return txt


def add_indefinite_article(role_name):
    """Check if a role name has a determinative adjective before it, and if not, add the correct one"""

    # Check if the first word is a determinative adjective
    determinative_adjectives = ["a", "an", "the"]
    words = role_name.split()
    if words[0].lower() not in determinative_adjectives:
        # Use "a" or "an" based on the first letter of the role name
        determinative_adjective = "an" if words[0][0].lower() in "aeiou" else "a"
        role_name = f"{determinative_adjective} {role_name}"

    return role_name


def generate_summary_and_answer(question, data, searcher, embedding_model, model,
                                max_new_tokens=2048, temperature=0.4, role="expert"):
    """Generate an answer for a given question using context from a dataset"""

    # Embed the input question using the provided embedding model
    embeded_question = np.array(get_embedding(question, embedding_model)).reshape(1, -1)

    print("Starting SCANN")
    start = time.time()
    # Find similar contexts in the dataset based on the embedded question
    neighbors, distances = searcher.search_batched(embeded_question)
    end = time.time()
    scann_time = end - start
    print(f"Time taken: {scann_time} seconds")
    # Extract context from the dataset based on the indices of similar contexts
    context = " ".join([data[pos] for pos in np.ravel(neighbors)])

    # Get the end-of-sentence token from the tokenizer
    try:
        EOS_TOKEN = model.tokenizer.eos_token
    except:
        EOS_TOKEN = "<eos>"

    # Add a determinative adjective to the role
    role = add_indefinite_article(role)
    #     print(context)

    #     # Generate a prompt for summarizing the context
    prompt = f"""
             Summarize this context: "{context}" in order to answer the question "{question}" as {role}\
             SUMMARY:
             """.strip() + EOS_TOKEN
    # Generate a summary based on the prompt
    #     print("Starting generating context summary")
    #     start = time.time()
    results = model.generate_text(prompt, max_new_tokens, temperature)
    #     end = time.time()
    #     prompt_time = end - start
    #     print(f"Time taken: {prompt_time} seconds")
    # Clean the generated summary
    summary = clean_text(results[0].split("SUMMARY:")[-1], EOS_TOKEN)

    # Generate a prompt for providing an answer
    prompt = f"""
             Here is the context: {summary}
             Using the relevant information from the context 
             and integrating it with your knowledge,
             provide an answer as {role} to the question: {question}.
             If the context doesn't provide
             any relevant information answer with 
             [I couldn't find a good match in my
             knowledge base for your question, 
             hence I answer based on my own knowledge]. Please give the answer within 50 words. \
             ANSWER:
             """.strip() + EOS_TOKEN

    print("Prompt:\n", prompt)

    print("Starting generating answer based on prompt")
    start = time.time()
    # Generate an answer based on the prompt
    results = model.generate_text(prompt, max_new_tokens, temperature)
    end = time.time()
    answer_time = end - start
    print(f"Time taken: {answer_time} seconds")

    # Clean the generated answer
    answer = clean_text(results[0].split("ANSWER:")[-1], EOS_TOKEN)

    # Return the cleaned answer
    return answer, answer_time, scann_time


class AIAssistant():
    """An AI assistant that interacts with users by providing answers based on a provided knowledge base"""

    def __init__(self, gemma_model, embeddings_name="thenlper/gte-large", temperature=0.4, role="expert"):
        """Initialize the AI assistant."""
        # Initialize attributes
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
        """Query the knowledge base of the AI assistant."""
        # Generate and print an answer to the query
        result, answer_time, scann_time = generate_summary_and_answer(query,
                                                                      self.knowledge_base,
                                                                      self.searcher,
                                                                      self.embedding_model,
                                                                      self.gemma_model,
                                                                      temperature=self.temperature,
                                                                      role=self.role)
        #         print(answer)
        return result, answer_time, scann_time

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


# Pre-compile the regular expression pattern for better performance
BRACES_PATTERN = re.compile(r'\{.*?\}|\}')


def remove_braces_and_content(text):
    """Remove all occurrences of curly braces and their content from the given text"""
    return BRACES_PATTERN.sub('', text)


def clean_string(input_string):
    """Clean the input string."""

    # Remove extra spaces by splitting the string by spaces and joining back together
    cleaned_string = ' '.join(input_string.split())

    # Remove consecutive carriage return characters until there are no more consecutive occurrences
    cleaned_string = re.sub(r'\r+', '\r', cleaned_string)

    # Remove all occurrences of curly braces and their content from the cleaned string
    cleaned_string = remove_braces_and_content(cleaned_string)

    # Return the cleaned string
    return cleaned_string


def load_dataset(url):
    return pd.read_parquet(url)


if __name__ == '__main__':
    text_corpus = load_dataset("hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet")
    question_answer = load_dataset("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
    # text_corpus = load_dataset("hf://datasets/rag-datasets/rag-mini-bioasq/data/passages.parquet/part.0.parquet")
    # question_answer = load_dataset("hf://datasets/rag-datasets/rag-mini-bioasq/data/test.parquet/part.0.parquet")
    questions = question_answer['question'].tolist()
    answers = question_answer['answer'].tolist()
    embeddings_name = "thenlper/gte-large"
    tokenizer = "/kaggle/input/gemma/gemmacpp/2b-it-sfp/4/tokenizer.spm"
    compressed_weights = "/kaggle/input/gemma/gemmacpp/2b-it-sfp/4/2b-it-sfp.sbs"
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

    # # # Map the intended knowledge base to embeddings and index it
    # gemma_ai_assistant.learn_knowledge_base(knowledge_base=knowledge_base)

    # # # Save the embeddings to disk (for later use)
    # gemma_ai_assistant.save_embeddings()

    # Uploading the knowledge base and embeddings to the AI assistant
    gemma_ai_assistant.store_knowledge_base(knowledge_base=knowledge_base)
    gemma_ai_assistant.load_embeddings(filename="data/embeddings.npy")
    # # Start the logger running in a background process. It will keep running until you tell it to stop.
    # # We will save the CPU and GPU utilisation stats to a CSV file every 0.2 seconds.
    # import subprocess
    # !rm -f log_compute.csv
    # logger_fname = 'log_compute.csv'
    # logger_pid = subprocess.Popen(
    #     ['python', 'log_gpu_cpu_stats.py',
    #      logger_fname,
    #      '--loop',  '30',  # Interval between measurements, in seconds (optional, default=1)
    #     ])
    # print('Started logging compute utilisation')

    answer_time_arr, scann_time_arr, prompt_time_arr = [], [], []
    i = 0
    for i, (question, answer) in enumerate(zip(questions, answers), i):
        print(f"Question #{i}")
        print("Question: " + question)
        print("Answer:" + answer)
        result, answer_time, scann_time = gemma_ai_assistant.query(question)
        answer_time_arr.append(answer_time)
        scann_time_arr.append(scann_time)
        # prompt_time_arr.append(prompt_time)
        print("Result:" + result)
        i += 1
        #     break
        if i == 15:
            break

    print(f"Avg scann time {statistics.fmean(scann_time_arr)}")
    print(f"Avg prompt time {statistics.fmean(prompt_time_arr)}")
    print(f"Avg answer time {statistics.fmean(answer_time_arr)}")
