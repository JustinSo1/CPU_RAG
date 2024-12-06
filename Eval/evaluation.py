import csv
import glob
import re
import os
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from api_call import API_CLIENT

# rag-mini-wikipedia
# file_name = '../gemma_answers_50_chunk_size_1_neighbors.csv'

map = {}
score_map = {}
counter_context_missing = 0
counter_context_exist = 0
total_score = 0

failures = [
    "The context does not",
    "I couldn't find a good match in my knowledge base",
    "I cannot answer this question"
]


def load_the_dataset():
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    return ds


def check_failures(input_string):
    for failure in failures:
        if failure in input_string:
            return True
    return False


def process_csv_and_call_api(file_name):
    global counter_context_exist
    global counter_context_missing
    # TODO: Change this to use pandas package
    with open(file_name, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            answer = row['Answer']
            id = row['']
            if check_failures(answer):
                counter_context_missing += 1
            else:
                counter_context_exist += 1
            map[int(id[1:]) - 1] = answer
    directory_path = f"responses/{file_name}"
    os.makedirs(directory_path, exist_ok=True)


def process_feedback(feedback, id, inputPrompt, userAnswer, correctAnswer, file_name):
    parts = feedback.split("[RESULT]")
    feedback_part = parts[0].strip()
    result_part = parts[1].strip()

    match = re.search(r'(\d+)$', result_part)
    result_num = match.group(1)

    # Output the result
    with open(f"responses/{file_name}/{id}.txt", "w") as file:
        file.write(f"[PROMPT]: {inputPrompt}\n")
        file.write(f"[USER_ANSWER]: {userAnswer}\n")
        file.write(f"[CORRECT_ANSWER]: {correctAnswer}\n")
        file.write(feedback_part + '\n')
        file.write(f"[RESULT]: {result_num}")
    # Return grade to average it
    return int(result_num)


if __name__ == '__main__':
    accuracy_dict = {}
    # rag-mini-wikipedia
    for file_name in glob.glob("../data/dataset/rag_wikipedia/results/chunking_neighbors/*.csv"):
        print(f"Processing {file_name}")
        process_csv_and_call_api(file_name)
        global_ds = load_the_dataset()
        api_obj = API_CLIENT()
        for index in map.keys():
            cur = global_ds['test'][index]
            inputPrompt = cur['question']
            userAnswer = map[index]
            correctAnswer = cur['answer']
            q_id = cur['id']
            print(f"Processing {index}")
            feedback = api_obj.send_query(inputPrompt, userAnswer, correctAnswer)
            curScore = process_feedback(feedback, q_id, inputPrompt, userAnswer, correctAnswer, file_name)
            score_map[curScore] = score_map.get(curScore, 0) + 1
            total_score += curScore
        print(f"Context missing: {counter_context_missing}")
        print(f"Content exist: {counter_context_exist}")
        print(f"Average Score: {total_score / (counter_context_missing + counter_context_exist)}")
        print(score_map)
        accuracy_dict[os.path.basename(file_name)] = {
            "missing_context": counter_context_missing,
            "exist_content": counter_context_exist,
            'average_score': total_score / (counter_context_missing + counter_context_exist),
            'score_map': score_map
        }
    df = pd.DataFrame.from_dict(accuracy_dict)
    df.to_csv('gemma_wiki_chunking_accuracy_stats.csv')
