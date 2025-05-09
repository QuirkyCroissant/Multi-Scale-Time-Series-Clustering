import pandas as pd
import numpy as np
from project_utilities import (import_dataframe_from_csv_indexed, 
                               deindex_dataframe, 
                               compute_and_save_accuracy, 
                               export_dataframe_to_csv)
import config
import os
import time

def interpolate_dataset(dataframe: pd.DataFrame, freq='D', method='linear', spline_order=3):
    '''Applies interpolation(linear/cubic splines) onto the given dataset depending on the parameters'''
    full_index = pd.date_range(start=pd.to_datetime(config.TIME_SERIES_START_DATE), 
                               periods=config.PERIOD_LENGTH, 
                               freq=freq)
    df_full = dataframe.reindex(full_index)
    
    if (method == 'krogh' or method == 'piecewise_polynomial' 
    or method == 'pchip' or method == 'akima' or method == 'cubicspline' 
    or method == 'polynomial' or method == 'spline'):
        df_repaired = df_full.interpolate(method=method, order=spline_order)

    elif (method == 'linear' or method == 'time' or method == 'index' or method == 'values' 
    or method == 'pad' or method == 'nearest' or method == 'zero' 
    or method == 'slinear' or method == 'quadratic' or method == 'cubic' 
    or method == 'barycentric'):
        df_repaired = df_full.interpolate(method=method)
    else:
        raise ValueError(f"Unsupported interpolation method. {method}")

    df_repaired.fillna(method="ffill", inplace=True)
    df_repaired.fillna(method="bfill", inplace=True)

    if df_repaired.isnull().any().any():
        raise ValueError(f"Interpolation with method '{method}' resulted in NaNs.")
    if not np.all(np.isfinite(df_repaired['value'].values)):
        raise ValueError(f"Interpolation with method '{method}' resulted in non-finite values.")

    df_repaired = df_repaired.reset_index().rename(columns={'index': 'time'})

    return df_repaired


def run_restoration_methods(dataframe: pd.DataFrame, series_id=0):
    '''triggers multiple interpolation methods and saves accuracy results'''
    for method_name in config.INTERPOLATION_METHODS:
        start = time.time()
        print(f"Interpolation with {method_name}: Started: Started for Series #{series_id}")
        df = dataframe.copy()
        # because of numerical instability of polynomial interpolation for methods such as 
        # "barycentric", "polynomial" and "krogh". The algorithm will try 3 times to build 
        # a restored dataset
        MAX_RETRIES = 3
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                repaired_df = interpolate_dataset(df, method=method_name)
                break  
            except ValueError as e:
                print(f"[Warning] Attempt {attempt}/{MAX_RETRIES} failed for method '{method_name}' on Series #{series_id}: {e}")
                if attempt == MAX_RETRIES:
                    print(f"[Skip] Interpolation failed permanently for '{method_name}' on Series #{series_id}")
                    repaired_df = None
                    break
                else:
                    continue

        # restoration failed, so it will stop the restoration for that file
        if repaired_df is None:
            continue

        time_difference = time.time() - start
        
        compute_and_save_accuracy(repaired_df, method_name, series_id, time_difference)

        export_dataframe_to_csv(repaired_df, 
                                filename=f"{config.SYN_EXPORT_DATA_NAME}_{series_id}_{method_name}", 
                                output_dir=os.path.join(config.TO_AGGREGATED_DATA_DIR, method_name))
        

        


def restore_time_series_data(is_demo_execution):
    '''Imports the corrupt dataset and moves it down the restoration stream. In following 
    functions data gets interpolated, up- and downsampled and than saved in the data 
    directory'''

    if is_demo_execution:
        corrupt_origin_dir = config.TO_CORRUPTED_DATA_DIR
    else:
        raise NotImplementedError("aggregation pipeline for production mode not implemented yet.")


    for i in range(config.AMOUNT_OF_INDIVIDUAL_SERIES):

        print(f"\nRestoring time series #{i}...")

        corrupt_data: pd.DataFrame = import_dataframe_from_csv_indexed(
            f"{config.SYN_EXPORT_DATA_NAME}_{i}_corrupted",
            input_dir=corrupt_origin_dir
            )
        
        run_restoration_methods(corrupt_data, series_id=i)
        print("\n")

    print("Completed restoration of all time series data")
