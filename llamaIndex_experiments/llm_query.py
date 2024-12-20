import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from llama_index.core import Settings
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.llama_cpp import LlamaCPP


def main():
    # os.environ["OPENAI_API_KEY"] = "random"
    # model_url = "https://huggingface.co/google/gemma-2-2b-it-GGUF/resolve/main/2b_it_v2.gguf"
    model_url = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q5_K_M.gguf"
    llm = create_llama_cpp_model(model_url)
    load_dotenv('../.env')

    # # Define llm parameters
    #llm = AzureChatOpenAI(
    #    deployment_name=os.environ['MODEL'],
    #    openai_api_version=os.environ['API_VERSION'],
    #    openai_api_key=os.environ['OPENAI_API_KEY'],
    #    azure_endpoint=os.environ['OPENAI_API_BASE'],
    #    openai_organization=os.environ['OPENAI_ORGANIZATION']
    #)
    llm_predictor = LangChainLLM(llm=llm)

    question_answer = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-wikipedia/data/test.parquet/part.0.parquet")
    questions = question_answer['question'].tolist()
    answers = question_answer['answer'].tolist()

    #Settings.llm = llm_predictor
    Settings.llm = llm
    answer_dict = {}
    for i, (question, answer) in enumerate(zip(questions, answers), 0):
        print(f"Question #{i}")
        print(f"Question: {question}")
        print(f"Ground Truth: {answer}")
        start_llm_response_time = time.perf_counter()
        response = llm_predictor.complete(question)
        end_llm_response_time = time.perf_counter()
        print(f"Result: {response}")
        answer_dict[f"Q{i}"] = {"Answer": response,
                                "llm_response_time": end_llm_response_time - start_llm_response_time,
                                }
#        if i == 50:
#            break
#        break
    df = pd.DataFrame.from_dict(answer_dict)
    print(df)
    df.to_csv("llama_index_wiki_no_rag_gemma-2-2b-it.csv")


def create_llama_cpp_model(model_url):
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        # model_path="../data/models/gemma2/gemma-2-2b-it.Q5_K_M.gguf",
        temperature=0.1,
        max_new_tokens=256,
        context_window=8192,
        # kwargs to pass to __call__()
        # generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"repeat_penalty": 1, "penalize_nl": False, "chat_format": "gemma"},
        # transform inputs into Llama2 format
        # messages_to_prompt=messages_to_prompt,
        # completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm


if __name__ == '__main__':
    log_file = f"logs/llamaindex_wiki_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.log"
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
