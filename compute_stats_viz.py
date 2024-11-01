import pickle

import pandas as pd
from matplotlib import pyplot as plt


def visualize_log_compute_file(logger_fname):
    logger_df = pd.read_csv(logger_fname)

    print("Avg CPU usage", logger_df['CPU (%)'].mean())
    print("Avg RAM usage", logger_df['RAM (%)'].mean())

    t = pd.to_datetime(logger_df['Timestamp (s)'], unit='s')
    cols = [col for col in logger_df.columns
            if 'time' not in col.lower() and 'temp' not in col.lower()]
    plt.figure(figsize=(15, 9))
    plt.plot(t, logger_df[cols])
    plt.legend(cols)
    plt.xlabel('Time')
    plt.ylabel('Utilisation (%)')
    plt.show()

    for col in logger_df.columns:
        if 'time' in col.lower(): continue
        plt.figure(figsize=(15, 9))
        plt.plot(t, logger_df[col])
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.show()


def convert_answers_to_csv(results_pickle, csv_fname):
    with open(f'{results_pickle}', 'rb') as f:
        x = pickle.load(f)
    res = {}
    for i in range(len(x)):
        res[f"Q{i + 1}"] = {"Answer": x[i]}
    # print(res)
    df = pd.DataFrame.from_dict(res, orient='index')
    print(df)
    df.to_csv(f"{csv_fname}")


if __name__ == '__main__':
    # logger_fname = "log_compute.csv"
    # visualize_log_compute_file(logger_fname)
    # convert_answers_to_csv('results.pickle', 'gemmaWikipediaAnswers.csv')
    df = pd.read_csv('llama_avg_rag_stats.csv')
    # print(df)
    df = df.transpose()
    df = df.rename(columns={0: 'avg_scann_time', 1: 'avg_summary_time', 2: 'avg_answer_time'})
    print(df)

    df.plot()
    plt.legend(title='Llama 3.2 3b RAG pipeline runtime vs number of threads')

    # get rid of the ticks between the labels - not necessary
    plt.xticks(ticks=range(0, len(df)))
    plt.ylabel("Runtime in seconds")

    plt.show()
