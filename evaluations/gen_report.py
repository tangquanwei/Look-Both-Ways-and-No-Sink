import json
import pandas as pd
from task_config import *
import argparse

def get_metric_path(task):
    if task in TASK_LIST_CLASSIFICATION:
        return ["scores", "test", 0, "en", "accuracy"]
    if task in TASK_LIST_PAIR_CLASSIFICATION:
        return ["scores", "test", 0, "en", "cos_sim", "precision", "cosine_precision"]
    if task in TASK_LIST_CLUSTERING:
        return ["scores", "test", 0, "en", "v_measure"]
    if task in TASK_LIST_RERANKING:
        return ["scores", "test", 0, "en", "map"]
    if task in TASK_LIST_RETRIEVAL:
        return ["scores", "test", 0, "en", "ndcg_at_10"]
    if task in TASK_LIST_STS:
        return ["scores", "test", 0, "en", "en-en", "cos_sim", "spearman", "cosine_spearman"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default="SUBSET")
    parser.add_argument('--model', default='.', type=str)

    args = parser.parse_args()
    
    tasks = (
        TASK_LIST if args.tasks == "ALL" else
        EASY_TASK_LIST if args.tasks == "EASY" else
        TASK_SUBSET if args.tasks == "SUBSET" else
        TASK_LIST_CLASSIFICATION if args.tasks == "Classification" else
        TASK_LIST_CLUSTERING if args.tasks == "Clustering" else
        TASK_LIST_PAIR_CLASSIFICATION if args.tasks == "PairClassification" else
        TASK_LIST_RERANKING if args.tasks == "Reranking" else
        TASK_LIST_RETRIEVAL if args.tasks == "Retrival" else
        TASK_LIST_STS if args.tasks == "STS" else []
    )

    results = pd.DataFrame()
    

    flag = True
    model = args.model
    for task in tasks:
        try:
            with open("{}/{}.json".format(model, task)) as f:
                result = json.load(f)
                for key in get_metric_path(task):
                    try:
                        result = result[key]
                    except:
                        continue
                results.loc[task, model] = result
        except FileNotFoundError:
            print(f"{model}/{task} file not found. Skipping...")
            results.loc[task, model] = -1
            flag = False
    if flag:
        results.loc["--Avg--"] = results.mean()
        avg_value = float(results.loc['--Avg--'].iloc[-1])
        print(f"{avg_value:.4f}")
    else:
        print(0)
    try:
        results.to_markdown(f'{model}/experiment_results_{args.tasks.lower()}.md')
    except:
        return


if __name__ == "__main__":
    main()