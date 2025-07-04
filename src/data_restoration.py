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
    '''Applies interpolation onto the given dataset depending on the parameters'''

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        if 'time' in dataframe.columns:
            dataframe['time'] = pd.to_datetime(dataframe['time'])
            dataframe.set_index('time', inplace=True)
        else:
            raise ValueError("Expected 'time' column for datetime index.")

    if not dataframe.index.is_unique:
        duplicates = dataframe.index.duplicated(keep='first')
        print(f"WARN: Dropped {duplicates.sum()} duplicate timestamps before reindexing.")
        dataframe = dataframe[~duplicates]

    full_index = pd.date_range(start=pd.to_datetime(config.TIME_SERIES_START_DATE), 
                               periods=config.PERIOD_LENGTH, 
                               freq=freq)
    df_full = dataframe.reindex(full_index)
    
    try:
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
    except Exception as e:
        print(f"Fallback-Interpolation with method '{method}' failed: {e}. Falling back to 'linear'.")
        df_repaired = df_full.interpolate(method='linear')

    df_repaired.fillna(method="ffill", inplace=True)
    df_repaired.fillna(method="bfill", inplace=True)

    if df_repaired.isnull().any().any():
        raise ValueError(f"Interpolation with method '{method}' resulted in NaNs.")
    if not np.all(np.isfinite(df_repaired['value'].values)):
        raise ValueError(f"Interpolation with method '{method}' resulted in non-finite values.")

    df_repaired = df_repaired.reset_index().rename(columns={'index': 'time'})

    return df_repaired


def run_restoration_methods(dataframe: pd.DataFrame, series_id=0, 
                            restore_method="all", 
                            mode="demo"):
    '''triggers multiple interpolation methods and saves accuracy results'''
    if restore_method == "all":
        chosen_methods = config.INTERPOLATION_METHODS
    else:
        chosen_methods = [restore_method]

    for method_name in chosen_methods:
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
                print(f"WARN: Attempt {attempt}/{MAX_RETRIES} failed for method '{method_name}' on Series #{series_id}: {e}")
                if attempt == MAX_RETRIES:
                    print(f"SKIP: Interpolation failed permanently for '{method_name}' on Series #{series_id}")
                    repaired_df = None
                    break
                else:
                    continue

        # restoration failed, so it will stop the restoration for that file
        if repaired_df is None:
            continue

        time_difference = time.time() - start
        
        if mode == "demo":
            compute_and_save_accuracy(repaired_df, method_name, series_id, time_difference)

        output_dir = os.path.join(config.TO_AGGREGATED_DATA_DIR, method_name)
        base_filename = config.SYN_EXPORT_DATA_NAME
        
        if mode == "prod":
            output_dir = os.path.join(output_dir, "prod")
            base_filename = config.PROD_EXPORT_DATA_NAME


        export_dataframe_to_csv(repaired_df, 
                                filename=f"{base_filename}_{series_id}_{method_name}", 
                                output_dir=output_dir)
        

        


def restore_time_series_data(restore_method, is_demo_execution):
    '''Imports the corrupt dataset and moves it down the restoration stream. In following 
    functions data gets interpolated, up- and downsampled and than saved in the data 
    directory'''

    if is_demo_execution:
        unrestored_data_dir = config.TO_CORRUPTED_DATA_DIR
        base_filename = config.SYN_EXPORT_DATA_NAME
        unrestored_suffix = "corrupted"
        mode = "demo"

    else:
        unrestored_data_dir = config.TO_PROD_SERIES_EXPORT_DATA_DIR
        base_filename = config.PROD_EXPORT_DATA_NAME
        amount = config.AMOUNT_OF_INDIVIDUAL_SERIES
        unrestored_suffix = "raw"
        mode = "prod"


    for i in range(config.AMOUNT_OF_INDIVIDUAL_SERIES):

        print(f"\nRestoring time series #{i}...")

        corrupt_data: pd.DataFrame = import_dataframe_from_csv_indexed(
            f"{base_filename}_{i}_{unrestored_suffix}",
            input_dir=unrestored_data_dir
            )
        
        run_restoration_methods(corrupt_data, series_id=i, restore_method=restore_method, mode=mode)
        print("\n")

    print("Completed restoration of all time series data")
