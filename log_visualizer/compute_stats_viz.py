import glob
import pickle

import pandas as pd
from matplotlib import pyplot as plt

import re
import os


def visualize_log_compute_file(logger_fname, i):
    logger_df = pd.read_csv(logger_fname)
    avg_cpu_usage = logger_df['CPU (%)'].mean()
    avg_ram_usage = logger_df['RAM (%)'].mean()
    print("Avg CPU usage", avg_cpu_usage)
    print("Avg RAM usage", avg_ram_usage)

    t = pd.to_datetime(logger_df['Timestamp (s)'], unit='s')
    cols = [col for col in logger_df.columns
            if 'time' not in col.lower() and 'temp' not in col.lower()]
    plt.figure(figsize=(15, 9))
    plt.plot(t, logger_df[cols])
    plt.legend(cols)
    plt.xlabel('Time')
    plt.ylabel('Utilisation (%)')
    plt.savefig(f"data/log_compute{i}")
    plt.close()

    return avg_cpu_usage, avg_ram_usage

    # for col in logger_df.columns:
    #     if 'time' in col.lower(): continue
    #     plt.figure(figsize=(15, 9))
    #     plt.plot(t, logger_df[col])
    #     plt.xlabel('Time')
    #     plt.ylabel(col)
    #     plt.show()


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


def visualize_rag_pipeline_stats(fname, model_name):
    df = pd.read_csv(fname)
    # print(df)
    df = df.transpose()
    df = df.rename(columns={0: 'avg_scann_time', 1: 'avg_summary_time', 2: 'avg_answer_time'})
    print(df)
    df.plot()
    plt.legend(title=f'{model_name} RAG pipeline runtime vs number of threads')
    # get rid of the ticks between the labels - not necessary
    plt.xticks(ticks=range(0, len(df)))
    plt.ylabel("Runtime in seconds")
    plt.show()


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


if __name__ == '__main__':
    # logger_fname = "data/archives/run3bio/log_compute-1.csv"
    # visualize_log_compute_file(logger_fname,33)
    for i, fname in enumerate(sorted(glob.glob("data/dataset/rag_wikipedia/results/*.pickle"), key=natural_keys),
                              start=1):
        _, chunk_size, neighbors = os.path.basename(fname).split("_")
        # print(chunk_size, neighbors.split(".")[0])
        convert_answers_to_csv(fname,
                               f"data/dataset/rag_wikipedia/results/gemma_answers_{chunk_size}_chunk_size"
                               f"_{neighbors.split('.')[0]}_neighbors.csv")
    # visualize_rag_pipeline_stats("llama_avg_rag_stats.csv", "Llama 3.2 3b")
    # avg_util_usage = {}
    # for i, fname in enumerate(sorted(glob.glob("data/archives/run3bio/*.csv"), key=natural_keys), start=1):
    #     print(i)
    #     avg_cpu_usage, avg_ram_usage = visualize_log_compute_file(fname, i)
    #     avg_util_usage[f"{-1}_threads"] = {
    #         "AVG_CPU_USAGE": avg_cpu_usage,
    #         "AVG_RAM_USAGE": avg_ram_usage
    #     }
    # df = pd.DataFrame.from_dict(avg_util_usage, orient='index')
    # print(df)
    # df.plot()
    # plt.legend(title='Gemma RAG pipeline utilization vs number of threads')
    # # get rid of the ticks between the labels - not necessary
    # plt.xticks(ticks=range(0, len(df)))
    # plt.ylabel("Avg %")
    # plt.savefig("gemma_avg_util_rate.png")
    # plt.show()
