import pandas as pd
from project_utilities import import_dataframe_from_csv_indexed, deindex_dataframe, compute_and_save_accuracy, export_dataframe_to_csv
import config
import os

def interpolate_dataset(dataframe: pd.DataFrame, freq='H', method='linear', spline_order=3):
    '''Applies interpolation(linear/cubic splines) onto the given dataset depending on the parameters'''
    full_index = pd.date_range(start=dataframe.index.min(), end=dataframe.index.max(), freq=freq)
    df_full = dataframe.reindex(full_index)
    
    if method == 'linear':
        df_repaired = df_full.interpolate(method=method)
    elif method == 'cubicspline':
        df_repaired = df_full.interpolate(method=method, order=spline_order)
    else:
        raise ValueError("Unsupported interpolation method. Use 'linear' or 'cubicspline'.")

    df_repaired.fillna(method="ffill", inplace=True)
    df_repaired.fillna(method="bfill", inplace=True)

    df_repaired = df_repaired.reset_index().rename(columns={'index': 'time'})

    return df_repaired


def run_restoration_methods(dataframe: pd.DataFrame):
    '''triggers multiple interpolation methods and saves accuracy results'''
    for m in config.INTERPOLATION_METHODS:
        print(f"Interpolation with {m}: Started")
        df = dataframe.copy()
        repaired_df = interpolate_dataset(df, method=m)
        compute_and_save_accuracy(repaired_df, m)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir,"..", "data", "restored")
        export_dataframe_to_csv(repaired_df, 
                                filename=config.SYN_EXPORT_DATA_NAME + '_' + m, 
                                output_dir=output_dir)
        

        


def restore_time_series_data():
    '''Imports the corrupt dataset and moves it down the restoration stream. In following 
    functions data gets interpolated, up- and downsampled and than saved in the data 
    directory'''
    corrupt_data: pd.DataFrame = import_dataframe_from_csv_indexed(config.SYN_EXPORT_DATA_NAME + '_corrupted')
    
    run_restoration_methods(corrupt_data)

    print("Completed restoration of time series data")
