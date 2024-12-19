import os
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from api_call import API_CLIENT


def load_the_dataset():
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    return ds


def process_feedback(feedback, question_id, input_prompt, user_answer, correct_answer, response_dir):
    parts = feedback.split("[RESULT]")
    feedback_part = parts[0].strip()
    result_part = parts[1].strip()

    match = re.search(r'(\d+)$', result_part)
    result_num = match.group(1)

    # Output the result
    with open(f"{response_dir}/{question_id}.txt", "w") as file:
        file.write(f"[PROMPT]: {input_prompt}\n")
        file.write(f"[USER_ANSWER]: {user_answer}\n")
        file.write(f"[CORRECT_ANSWER]: {correct_answer}\n")
        file.write(feedback_part + '\n')
        file.write(f"[RESULT]: {result_num}")
    # Return grade to average it
    return int(result_num)


def count_context_cases(df):
    # Unreliable metric
    failures = [
        "The context does not",
        "I couldn't find a good match in my knowledge base",
        "I cannot answer this question"
    ]
    counter_context_missing = df.str.count('|'.join(failures)).sum()
    counter_context_exist = df.shape[0] - counter_context_missing
    return counter_context_exist, counter_context_missing


def create_question_answer_df(file_name):
    df = pd.read_csv(file_name, index_col=0)
    df = df.transpose()
    df = df['Answer']
    return df


def main():
    # rag-mini-wikipedia
    # file_names = glob.glob("../data/dataset/rag_wikipedia/results/chunking_neighbors/*.csv")
    experiment_name = 'llama_index_wiki_gemma-2-2b-it_accuracy_stats'
    file_names = ['llama_index_wiki_gemma-2-2b-it-Q5_K_M.csv']
    for file_name in file_names:
        print(f"Processing {file_name}")

        # Create response directory
        filename_without_ext = Path(file_name).stem
        directory_path = f"responses/{filename_without_ext}"
        os.makedirs(directory_path, exist_ok=True)

        # Parse csv
        df = create_question_answer_df(file_name)
        counter_context_exist, counter_context_missing = count_context_cases(df)

        # Load dataset questions and answers
        global_ds = load_the_dataset()
        questions = global_ds['test']['question']
        correct_answers = global_ds['test']['answer']
        user_answers = df.tolist()

        api_obj = API_CLIENT()

        score_map = {}
        total_score = 0
        accuracy_dict = {}

        for question_id, (question, correct_answer, user_answer) in enumerate(
                zip(questions, correct_answers, user_answers),
                start=1):
            print(f"Question ID: {question_id}, Question: {question}, CA: {correct_answer}, UA: {user_answer}")
            feedback = api_obj.send_query(question, user_answer, correct_answer)
            curScore = process_feedback(feedback, question_id, question, user_answer, correct_answer, directory_path)
            score_map[curScore] = score_map.get(curScore, 0) + 1
            total_score += curScore
#            break

        average_score = total_score / (counter_context_missing + counter_context_exist)

        print(f"Context missing: {counter_context_missing}")
        print(f"Content exist: {counter_context_exist}")
        print(f"Average Score: {average_score}")
        print(score_map)

        accuracy_dict[os.path.basename(file_name)] = {
            "missing_context": counter_context_missing,
            "exist_content": counter_context_exist,
            'average_score': average_score,
            'score_map': dict(score_map)
        }
    df = pd.DataFrame.from_dict(accuracy_dict)
    df.to_csv(f'{experiment_name}.csv')


if __name__ == '__main__':
    main()
