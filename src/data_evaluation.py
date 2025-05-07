import numpy as np
import pandas as pd
import os
import re
import json
from collections import defaultdict
from datetime import datetime

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

    evaluated_dataframe["MAPE (Mean ± Std)"] = evaluated_dataframe.apply(
        lambda row: format_mean_std(row["mape_mean"], row["mape_std"]), 
        axis=1
    )
    evaluated_dataframe["MSE (Mean ± Std)"] = evaluated_dataframe.apply(
        lambda row: format_mean_std(row["mse_mean"], row["mse_std"]), 
        axis=1
    )

    df_pretty = evaluated_dataframe[["method", "MAPE (Mean ± Std)", "MSE (Mean ± Std)"]].sort_values("method")
    return df_pretty


def evaluate_restoration_results(metrics):
    '''Searches for the latest log files, per interpolation method and aggregates the values into several output 
    tables to visualise interpolation performance per method, those tables will get exported into dedicated directories
    to save them after the runtime.'''

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
    export_dataframe_to_csv(df=df_results, 
                            filename=f"{config.RESTORATION_RAW_TABLE_EXPORT_NAME}_{timestamp}", 
                            output_dir=config.TO_EVAL_INTERPOLATION_DIR,
                            eval=True)
    
    df_humanreadable = make_interpolation_df_humanreadable(df_results)
    print(df_humanreadable)
    export_dataframe_to_csv(df=df_humanreadable, 
                            filename=f"{config.RESTORATION_PRETTY_TABLE_EXPORT_NAME}_{timestamp}", 
                            output_dir=config.TO_EVAL_INTERPOLATION_DIR,
                            eval=True)
    
    #convert_dataframe_into_latex(df_humanreadable)

    print("Evaluation of Interpolation results concluded.")
            



    


def initialise_specific_evaluation(mode: str, metrics=[]):
    '''TODO: TBD'''
    if mode == "aggregation":
        check_metrics(mode, metrics)
        evaluate_restoration_results(metrics)
        
    elif mode == "clustering_demo":
        check_metrics(mode, metrics)
        pass
    elif mode == "clustering_prod":
        check_metrics(mode, metrics)
        pass
    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}")