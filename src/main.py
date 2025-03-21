import argparse
from data_generation import create_time_series, convert_time_series_to_dataframe
from project_utilities import plot_time_series, export_dataframe_to_csv
import config

import numpy as np

def run_prototype():
    '''function which triggers prototyp mode of application,
        which consists of generating a synthetic heterogeneous
        dataset, preprocesses it accordingly, and than moving
        it further upstream for clustering.'''
    print("Running Application in Prototype mode:")
    print("Triggering generation of synthetic dataset")
    
    ts_data = create_time_series(config.TS_META)
    plot_time_series(config.TIME, ts_data, title=config.SYN_EXPORT_TITLE, 
                                       xlabel='Time(Days)', ylabel='Value')
    
    print("Converting time series to dataframe and exporting to CSV")
    syn_ts_df = convert_time_series_to_dataframe(config.TIME, ts_data)
    export_dataframe_to_csv(syn_ts_df)
    

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