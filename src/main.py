import argparse
from data_generation import create_time_series, convert_time_series_to_dataframe
from data_corruption import corrupt_time_series_data
from data_restoration import restore_time_series_data
from data_clustering import initiate_clustering_process
from project_utilities import ( plot_time_series, export_dataframe_to_csv, 
                               import_dataframe_from_csv, import_dataframe_from_csv_indexed,
                               deindex_dataframe, plot_time_series_comparison )
import config

import numpy as np
import pandas as pd
import os


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

def aggregation_pipeline(activate_restoration=False):
    '''aggregation pipeline'''
    if activate_restoration:
        restore_time_series_data()
    

def clustering_pipeline(comp_dist=False):
    '''clustering pipeline, which consists of 2 parts(distance computation and clustering). 
    Included distance argument decides if dissimilarity is computed or an already exported 
    matrix is used for the subsequent clustering.'''
    initiate_clustering_process(comp_dist)
    
    

def run_prototype(generate_data, 
                  restore=False, 
                  plot=False, 
                  compute_dist=False
                  ):
    '''function which triggers prototyp mode of application,
        which consists of generating a synthetic heterogeneous
        dataset, preprocesses it accordingly, and than moving
        it further upstream for clustering.'''
    print("Running Application in Prototype mode:")

    if generate_data:
        print("Triggering generation of synthetic dataset")
        
        demo_generation_pipeline()

        print("Triggering corruption of synthetic dataset")
        demo_corruption_pipeline()
    
    if plot:
        print("Generates comparisson plot of time series data")

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
        del time_series_dict
    
    print("Triggering Aggregation Pipeline")
    aggregation_pipeline(restore)

    print("Triggering Clustering Pipeline")
    clustering_pipeline(comp_dist=compute_dist)
    


    if plot:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(script_dir,"..", "data", "restored")

        clean_df = import_dataframe_from_csv(config.SYN_EXPORT_DATA_NAME+"_clean")
        for method in config.INTERPOLATION_METHODS:
            
            aggregated_df = import_dataframe_from_csv(
                config.SYN_EXPORT_DATA_NAME+"_"+method, 
                input_dir=input_dir)
            
            time_series_dict = {
                "Clean_TS": (clean_df["Time"], clean_df["Value"]),
                f"{method}_TS": (aggregated_df["time"], aggregated_df["value"])
                }
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "..", "experiments", "plots", "interpolations")

            plot_time_series_comparison(
                time_series_dict, 
                title=f"{method}-Interpolation_Comparison",
                output_dir=output_dir
                )
            del time_series_dict, aggregated_df



        
        
    
    

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
        required=True,
        choices=["demo", "prod"],
        help="Select the mode: 'demo' for synthetic dataset generation and testing, 'prod' for processing the pre-specified dataset."
    )

    parser.add_argument(
        "--new_data",
        action="store_true",
        help="Include this flag (only in demo mode) to generate corrupted, synthetic data."
    )

    parser.add_argument(
        "--comp_img",
        action="store_true",
        help="Saves comparisson plot of synthetic time series and its different versions downstream. (experiments/plots)"
    )

    parser.add_argument(
        "--restore",
        action="store_true",
        help="Include this flag to aggregate, interpolate and save faulty " \
        "input data that will be used for clustering. (data/restored)"
    )

    parser.add_argument(
        "--dist",
        action="store_true",
        help="Include this flag to compute and save the dissimilarity measure. (experiments/distance_matrices)"
    )

    args = parser.parse_args()

    if args.mode != "demo" and args.new_data:
        parser.error("The --new_data flag is only allowed when --mode is set to demo.")

    if args.mode == "demo":
        run_prototype(
            generate_data=args.new_data, 
            restore=args.restore,
            compute_dist=args.dist,
            plot=args.comp_img
        )
    elif args.mode == "prod":
        run_final()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()