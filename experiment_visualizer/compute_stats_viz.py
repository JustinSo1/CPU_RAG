import collections
import pickle
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm


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


def plottable_3d_info(df: pd.DataFrame):
    """
    Transform Pandas data into a format that's compatible with
    Matplotlib's surface and wireframe plotting.
    """
    index = df.index
    columns = df.columns

    x, y = np.meshgrid(np.arange(len(columns)), np.arange(len(index)))
    z = np.array([[df[c][i] for c in columns] for i in index])

    xticks = dict(ticks=np.arange(len(columns)), labels=columns)
    yticks = dict(ticks=np.arange(len(index)), labels=index)

    return x, y, z, xticks, yticks


def visualize_chunk_size_experiment_stats(fname, model_name):
    df = pd.read_csv(fname)

    chunk_stats = collections.defaultdict(dict)
    for col in df.columns:
        arr = col.split("_")
        chunk_size, neighbors = arr[0], arr[3]
        chunk_stats[chunk_size][neighbors] = df[col].sum()

    stats_df = pd.DataFrame.from_dict(chunk_stats)

    # Transform to Matplotlib friendly format.
    x, y, z, xticks, yticks = plottable_3d_info(stats_df)

    # Set up axes and put data on the surface.
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    axes.plot_surface(x, y, z)

    # Customize labels and ticks (only really necessary with
    # non-numeric axes).
    axes.set_xlabel('Chunk Size')
    axes.set_ylabel('Neighbors')
    axes.set_zlabel('Runtime')
    axes.set_zlim3d(bottom=0)
    plt.xticks(**xticks)
    plt.yticks(**yticks)
    plt.savefig("gemma_chunk_size_exp")
    plt.show()


def visualize_rag_pipeline_tokens_and_timings(input_fname, output_time_fname, output_token_fname):
    # fname = 'llamaIndex_experiments/llama_index_wiki_gemma-2-2b-it-Q5_K_M.csv'
    df = pd.read_csv(input_fname, index_col=0)
    # print(df)
    df = df.transpose()
    # df = df.drop('Answer', axis=1)
    time_df = df[['retrieval_time', 'llm_response_time']]
    time_df = time_df.astype(float)
    # print(time_df)
    time_df.plot(title="Wiki RAG Pipeline times", xlabel='Questions', ylabel='Time (s)')
    plt.tight_layout()
    plt.savefig(f"{output_time_fname}.png", dpi=300)
    # plt.show()
    token_df = df[['embedding_tokens', 'llm_prompt_tokens', 'llm_completion_tokens', 'total_llm_token_count']]
    token_df = token_df.astype(float)
    # print(token_df)
    token_df.plot(title="Wiki RAG Pipeline tokens", xlabel='Questions', ylabel='# of tokens')
    plt.tight_layout()
    plt.savefig(f"{output_token_fname}.png", dpi=300)


def visualize_rag_pipeline_timings(input_fname, output_time_fname):
    df = pd.read_csv(input_fname, index_col=0)
    # print(df)
    df = df.transpose()
    # df = df.drop('Answer', axis=1)
    time_df = df['llm_response_time']
    time_df = time_df.astype(float)
    # print(time_df.mean())
    time_df.plot(title="Wiki LLM times", xlabel='Questions', ylabel='Time (s)')
    plt.tight_layout()
    plt.savefig(f"{output_time_fname}.png", dpi=300)


if __name__ == '__main__':
    # logger_fname = "data/archives/run3bio/log_compute-1.csv"
    # visualize_log_compute_file(logger_fname,33)
    # for i, fname in enumerate(sorted(glob.glob("data/dataset/rag_wikipedia/results/*.pickle"), key=natural_keys),
    #                           start=1):
    #     _, chunk_size, neighbors = os.path.basename(fname).split("_")
    #     # print(chunk_size, neighbors.split(".")[0])
    #     convert_answers_to_csv(fname,
    #                            f"data/dataset/rag_wikipedia/results/gemma_answers_{chunk_size}_chunk_size"
    #                            f"_{neighbors.split('.')[0]}_neighbors.csv")
    # visualize_rag_pipeline_stats("llama_avg_rag_stats.csv", "Llama 3.2 3b")
    # visualize_chunk_size_experiment_stats("gemma_chunk_size_neighbor_experiment.csv", "Gemma 2b-it")
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
    # visualize_rag_pipeline_tokens_and_timings("llamaIndex_experiments/llama_index_wiki_gemma-2-2b-it-Q5_K_M.csv",
    #                               "data/dataset/rag_wikipedia/results/full_wiki/gemma2-2b-it-Q5_K_M_wiki_time",
    #                               "data/dataset/rag_wikipedia/results/full_wiki/gemma2-2b-it-Q5_K_M_wiki_tokens")
    # visualize_rag_pipeline_tokens_and_timings("llamaIndex_experiments/llama_index_wiki_gpt-4o.csv",
    #                                           "data/dataset/rag_wikipedia/results/full_wiki/gpt4o_wiki_time",
    #                                           "data/dataset/rag_wikipedia/results/full_wiki/gpt4o_wiki_tokens")
    # visualize_rag_pipeline_timings("llamaIndex_experiments/llama_index_wiki_no_rag_gemma-2-2b-it.csv",
    #                                "data/dataset/rag_wikipedia/results/full_wiki/gemma-2-2b-it_wiki_llm_only_time")
    # visualize_rag_pipeline_timings("llamaIndex_experiments/llama_index_wiki_no_rag_gpt-4o.csv",
    #                                "data/dataset/rag_wikipedia/results/full_wiki/gemma-2-2b-it_wiki_llm_only_time")
    visualize_rag_pipeline_tokens_and_timings("llamaIndex_experiments/llama_index_wiki_gpt-4o.csv",
                                              "data/dataset/rag_wikipedia/results/full_wiki/gpt4o_wiki_time",
                                              "data/dataset/rag_wikipedia/results/full_wiki/gpt4o_wiki_tokens")
