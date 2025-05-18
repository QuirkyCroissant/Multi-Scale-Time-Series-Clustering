import numpy as np
import pandas as pd
import os
import re
import json
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, rand_score, normalized_mutual_info_score

import config
from project_utilities import export_dataframe_to_csv

def check_metrics(mode, metrics):
    '''Checks if retrieved metrics are supported by the given'''
    if mode == "aggregation":
        allowed_metrics = config.INTERPOLATION_METRICS
    elif mode == "clustering_demo":
        allowed_metrics = config.CLUSTERING_EXTERNAL_METRICS
    else:
        allowed_metrics = config.CLUSTERING_INTERNAL_METRICS
    
    for metric in metrics:
        if metric not in allowed_metrics:
            raise ValueError(f"Unsupported metric: {metric}")
        
def make_interpolation_df_humanreadable(evaluated_dataframe):
    '''Helper function which refactors interpolation table format into rounded values 
    and unions averaged and standard deviated values into one cell, and therefore more readable. '''

    def format_mean_std(mean, std, decimals=4):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

    df_pretty = evaluated_dataframe[["method"]].copy()

    if "mape_mean" in evaluated_dataframe.columns and "mape_std" in evaluated_dataframe.columns:
        df_pretty["MAPE (Mean ± Std)"] = evaluated_dataframe.apply(
            lambda row: format_mean_std(row["mape_mean"], row["mape_std"]), 
            axis=1
        )

    if "mse_mean" in evaluated_dataframe.columns and "mse_std" in evaluated_dataframe.columns:
        df_pretty["MSE (Mean ± Std)"] = evaluated_dataframe.apply(
            lambda row: format_mean_std(row["mse_mean"], row["mse_std"]), 
            axis=1
        )

    return df_pretty.sort_values("method")


def evaluate_restoration_results(metrics):
    '''Searches for the latest log files, per interpolation method and aggregates the values into several output 
    tables to visualise interpolation performance per method, those tables will get exported into dedicated directories
    to save them after the runtime.'''

    # export metrics filtering using regex patterns
    pattern = '|'.join(map(re.escape,metrics))
    export_regex = re.compile(pattern, flags=re.IGNORECASE) 

    results = []
    methods = config.INTERPOLATION_METHODS
    log_date_pattern = re.compile(r'accuracy_log_ts_\d+_[\w]+_(\d{4}-\d{2}-\d{2})\.json')
    
    def get_latest_date(log_dir):
        dates = []
        for fname in os.listdir(log_dir):
            match = log_date_pattern.match(fname)
            if match:
                dates.append(match.group(1))
        if not dates:
            return None
        return max(dates)
    
    for method in methods:
        method_log_dir = os.path.join(config.TO_INTERPOLATION_LOGS_DIR, method)
        latest_date = get_latest_date(method_log_dir)
        if latest_date is None:
            print(f"{method}: no log files found")
            continue
        print(f"{method}: latest date is {latest_date}")

        mape_scores = []
        mse_scores = []
        
        for fname in os.listdir(method_log_dir):
            if latest_date not in fname:
                continue
            filepath = os.path.join(method_log_dir, fname)
            with open(filepath, 'r') as f:
                log = json.load(f)
                mape_scores.append(log['mapeScore'])
                mse_scores.append(log['mseScore'])
        
        results.append({
            "method": method,
            "mape_mean": np.mean(mape_scores),
            "mape_std": np.std(mape_scores),
            "mse_mean": np.mean(mse_scores),
            "mse_std": np.std(mse_scores),
        })

    timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    df_results = pd.DataFrame(results).sort_values("method")
    filtered_columns = [col for col in df_results.columns if export_regex.search(col)]
    df_results = df_results[["method"] + filtered_columns]

    export_dataframe_to_csv(df=df_results, 
                            filename=f"{config.RESTORATION_RAW_TABLE_EXPORT_NAME}_{timestamp}", 
                            output_dir=config.TO_EVAL_INTERPOLATION_DIR,
                            eval=True)
    
    df_humanreadable = make_interpolation_df_humanreadable(df_results)
    filtered_columns_readable = [col for col in df_humanreadable.columns if export_regex.search(col)]
    df_humanreadable = df_humanreadable[["method"] + filtered_columns_readable]
    print(df_humanreadable)
    export_dataframe_to_csv(df=df_humanreadable, 
                            filename=f"{config.RESTORATION_PRETTY_TABLE_EXPORT_NAME}_{timestamp}", 
                            output_dir=config.TO_EVAL_INTERPOLATION_DIR,
                            eval=True)

    print("Evaluation of Interpolation results concluded.")

def parse_filename_to_metadata(filename):
    base = filename.replace(".json", "")

    timestamp_match = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", base)
    if not timestamp_match:
        raise ValueError(f"Invalid log file format(timestamp): {base}")
    
    timestamp = timestamp_match.group(1)

    parts = base.split("_")

    try:
        method = parts[1]
        distance = parts[2]
    except IndexError:
        raise ValueError(f"Invalid log file format, missing parts(method/distance): {base}")
    

    radius = None
    if "fastDTW" in distance:
        radius_match = re.search(r"r(\d+)", base)
        radius = int(radius_match.group(1)) if radius_match else None
    
    normalized = "_n_" in base

    
    return {
        "method": method,
        "distance": distance,
        "radius": radius,
        "normalized": normalized,
        "timestamp": timestamp
    }


def import_latest_clustering_logs(log_dir):
    '''Collects most recent log files of either the default or the graph log directory
    and returns it in a 2D dictionary, for further processing'''
    
    latest_logs = {}
    for fname in os.listdir(log_dir):
        if not fname.endswith(".json"):
            continue
        try:
            metadata = parse_filename_to_metadata(fname)
            key = f"{metadata['method']}_{metadata['distance']}_{'n' if metadata['normalized'] else 'raw'}"
            curr_timeseries = metadata["timestamp"]
            
            if key not in latest_logs or curr_timeseries > latest_logs[key]["metadata"]["timestamp"]:
                with open(os.path.join(log_dir, fname), 'r') as f:
                    log = json.load(f)
                metadata["timestamp"] = curr_timeseries
                latest_logs[key] = {
                    "log": log,
                    "metadata": metadata,
                    "filename": fname
                }
        except Exception as e:
            print(f"Could not parse {fname}: {e}")

    return latest_logs

            
def compute_similarity_of_clustering_methodologies(traditional_results, graph_results):
    '''Collects default and graph results, computes external metrics and combines the 
    newfound information into a table, which gets returned in the end'''
    results = []
    

    for default_key, default_entry in traditional_results.items():
        base_log = default_entry["log"]
        
        base_labels = base_log["labels"]
        base_k = base_log["nClusters"]
        base_qty = base_log["dataQuantity"]
        
        for graph_key, graph_entry in graph_results.items():
            
            graph_log = graph_entry["log"]
            
            if graph_log["dataQuantity"] != base_qty:
                continue
            
            table_row = {
                "Graph_Clustering": graph_log["clusteringMethod"],
                "Graph_Dissimilarity": graph_log["distanceMeasure"],
                "Baseline_Method": base_log["clusteringMethod"],
                "Baseline_Dissimilarity": base_log["distanceMeasure"],
                "Normalized": default_entry["metadata"]["normalized"],
                "k": base_k,
                "ARI": adjusted_rand_score(base_labels, graph_log["labels"]),
                "RAND": rand_score(base_labels, graph_log["labels"]),
                "NMI": normalized_mutual_info_score(base_labels, graph_log["labels"])
            }

            results.append(table_row)

    return pd.DataFrame(results)


def evaluate_clustering_results(metrics):
    '''Starts the evaluation process for the clustering part of the project. Triggers and forwards all
    relevant parts to process the corresponding log files and aggregate them to a join table, which gets
    exported in the end.'''

    traditional_results = import_latest_clustering_logs(config.TO_DEFAULT_CLUSTERING_LOGS_DIR)
    graph_results = import_latest_clustering_logs(config.TO_GRAPH_CLUSTERING_LOGS_DIR)
    
    df_clustering_comparison = compute_similarity_of_clustering_methodologies(traditional_results, graph_results)
    print(df_clustering_comparison)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    export_dataframe_to_csv(df=df_clustering_comparison, 
                            filename=f"{config.CLUSTERING_RAW_TABLE_EXPORT_NAME}_{timestamp}", 
                            output_dir=config.TO_EVAL_CLUSTERING_DIR,
                            eval=True)

    


def initialise_specific_evaluation(mode: str, metrics=[]):
    '''Initiates evaluation mode for either the interpolation or clustering, based on passed program arguments'''
    if mode == "aggregation":
        check_metrics(mode, metrics)
        evaluate_restoration_results(metrics)
        
    elif mode == "clustering_demo":
        check_metrics(mode, metrics)
        evaluate_clustering_results(metrics)

    elif mode == "clustering_prod":
        check_metrics(mode, metrics)
        pass
    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}")