import re
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# import transformers
# from transformers import (AutoModelForCausalLM,
#                           AutoTokenizer,
#                           BitsAndBytesConfig,
#                          )


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
    print("Starting generating context summary")
    start = time.time()
    results = model.generate_text(prompt, max_new_tokens, temperature)
    end = time.time()
    prompt_time = end - start
    print(f"Time taken: {prompt_time} seconds")
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
    return answer, answer_time, scann_time, prompt_time


# Pre-compile the regular expression pattern for better performance
BRACES_PATTERN = re.compile(r'\{.*?}|}')


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


def read_csv(url):
    return pd.read_parquet(url)
