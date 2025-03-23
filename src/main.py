import argparse
from data_generation import create_time_series, convert_time_series_to_dataframe
from data_corruption import corrupt_time_series_data
from project_utilities import plot_time_series, export_dataframe_to_csv, import_dataframe_from_csv, import_dataframe_from_csv_indexed
from project_utilities import deindex_dataframe, plot_time_series_comparison
import config

import numpy as np
import pandas as pd


def demo_generation_pipeline():
    '''synthetic data generation pipeline'''
    ts_data = create_time_series(config.TS_META)
    plot_time_series(config.TIME, ts_data, title=config.SYN_EXPORT_TITLE, 
                                       xlabel='Time(Days)', ylabel='Value')
    
    print("Converting time series to dataframe and exporting to CSV")

    syn_ts_df = convert_time_series_to_dataframe(config.TIME, ts_data)
    export_dataframe_to_csv(syn_ts_df, config.SYN_EXPORT_DATA_NAME+"_clean")

def demo_corruption_pipeline():
    '''corrupted data generation pipeline'''
    clean_df = import_dataframe_from_csv_indexed(config.SYN_EXPORT_DATA_NAME+"_clean")
    corr_df = corrupt_time_series_data(clean_df)
    corr_df = deindex_dataframe(corr_df)
    export_dataframe_to_csv(corr_df, config.SYN_EXPORT_DATA_NAME+"_corrupted")


def run_prototype():
    '''function which triggers prototyp mode of application,
        which consists of generating a synthetic heterogeneous
        dataset, preprocesses it accordingly, and than moving
        it further upstream for clustering.'''
    print("Running Application in Prototype mode:")
    print("Triggering generation of synthetic dataset")
    
    demo_generation_pipeline()

    print("Triggering corruption of synthetic dataset")
    demo_corruption_pipeline()
    
    clean_df = import_dataframe_from_csv(config.SYN_EXPORT_DATA_NAME+"_clean")
    corr_df = import_dataframe_from_csv(config.SYN_EXPORT_DATA_NAME+"_corrupted")

    clean_df["Time"] = pd.to_datetime(clean_df["Time"])
    corr_df["time"] = pd.to_datetime(corr_df["time"])

    # Create a dictionary with one key-value pair:
    # The key is a label, and the value is a tuple of the time and value columns.
    time_series_dict = {
        "Clean_TS": (clean_df["Time"], clean_df["Value"]),
        "Corrupted_TS": (corr_df["time"], corr_df["value"])
        }

    plot_time_series_comparison(time_series_dict)
    
    

def run_final():
    '''production ready implementation which will be 
       implemented after an sufficient prototyp'''
    print("Running Application in Production mode:")
    pass

def main():

    parser = argparse.ArgumentParser(
        description="Main Script for Time Series Clustering"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "prod"],
        help="Select the mode: 'demo' for synthetic dataset generation and testing, 'prod' for processing the pre-specified dataset."
    )

    args = parser.parse_args()

    if args.mode == "demo":
        run_prototype()
    elif args.mode == "prod":
        run_final()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()