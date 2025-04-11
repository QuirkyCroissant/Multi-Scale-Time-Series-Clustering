import argparse
from data_generation import create_multiple_time_series, convert_time_series_to_dataframe
from data_corruption import corrupt_time_series_data
from data_restoration import restore_time_series_data
from data_clustering import start_clustering_pipeline
from project_utilities import ( plot_time_series, export_dataframe_to_csv, 
                               import_dataframe_from_csv, import_dataframe_from_csv_indexed,
                               deindex_dataframe, plot_time_series_comparison, traverse_to_method_dir)
import config

import numpy as np
import pandas as pd


def demo_generation_pipeline():
    '''synthetic data generation pipeline'''
    series_matrix = create_multiple_time_series()
    for i in range(series_matrix.shape[0]):
        plot_time_series(
            y=series_matrix[i], 
            title=f"{config.SYN_EXPORT_TITLE}_{i}_clean", 
            xlabel='Days', 
            ylabel='Value'
        )
    
    print("Converting time series to dataframes and exporting to CSV")

    syn_ts_dfs = [
        convert_time_series_to_dataframe(series_matrix[i]) 
        for i in range(series_matrix.shape[0])
    ]

    [
        export_dataframe_to_csv(syn_ts_dfs[i], 
                                filename= f"{config.SYN_EXPORT_DATA_NAME}_{i}", 
                                clean=True, output_dir=config.TO_CLEAN_DATA_DIR) 
        for i in range(len(syn_ts_dfs)) 
    ]

def demo_corruption_pipeline():
    '''Corrupts each individual clean time series and saves the result.'''
    for i in range(config.AMOUNT_OF_INDIVIDUAL_SERIES):
        clean_df = import_dataframe_from_csv_indexed(f"{config.SYN_EXPORT_DATA_NAME}_{i}_clean")

        corr_df = corrupt_time_series_data(clean_df)
        corr_df = deindex_dataframe(corr_df)

        export_dataframe_to_csv(corr_df, 
                                filename=f"{config.SYN_EXPORT_DATA_NAME}_{i}", 
                                corrupted=True, 
                                output_dir=config.TO_CORRUPTED_DATA_DIR
                                )
        
        plot_time_series(
            y=corr_df, 
            title=f"{config.SYN_EXPORT_TITLE}_{i}_corrupted", 
            xlabel='Days', 
            ylabel='Value',
            output_dir=config.TO_CORRUPTED_DATA_PLOTS_DIR
        )

    print("All time series have been corrupted and exported.")

def aggregation_pipeline(is_demo_execution, activate_restoration=False):
    '''aggregation pipeline'''
    if activate_restoration:
        restore_time_series_data(is_demo_execution)
    

def clustering_pipeline(comp_dist=False, normalize=False):
    '''clustering pipeline, which consists of 2 parts(distance computation and clustering). 
    Included distance argument decides if dissimilarity is computed or an already exported 
    matrix is used for the subsequent clustering.'''
    start_clustering_pipeline(comp_dist, normalize)
    
    

def run_prototype(generate_data, 
                  restore=False, 
                  compute_dist=False,
                  normalize=False,
                  plot=False
                  ):
    '''function which triggers prototyp mode of application,
        which consists of generating a synthetic heterogeneous
        dataset, preprocesses it accordingly, and than moving
        it further upstream for clustering.'''
    print("Running Application in Prototype mode:")

    is_demo_execution = True

    if generate_data:
        print("Triggering generation of synthetic dataset")
        
        demo_generation_pipeline()

        print("Triggering corruption of synthetic dataset")
        demo_corruption_pipeline()
    
    # plots the corrupt to clean time series plots
    if plot:
        print("Generating comparison plots for all synthetic time series data")

        for i in range(config.AMOUNT_OF_INDIVIDUAL_SERIES):

            clean_df = import_dataframe_from_csv(f"{config.SYN_EXPORT_DATA_NAME}_{i}_clean",
                                                 input_dir=config.TO_CLEAN_DATA_DIR)
            corr_df = import_dataframe_from_csv(f"{config.SYN_EXPORT_DATA_NAME}_{i}_corrupted",
                                                input_dir=config.TO_CORRUPTED_DATA_DIR)

            clean_df["time"] = pd.to_datetime(clean_df["time"])
            corr_df["time"] = pd.to_datetime(corr_df["time"])

            # Create a dictionary with one key-value pair:
            # The key is a label, and the value is a tuple of the time and value columns.
            time_series_dict = {
                f"Clean_TS_{i}": (clean_df["time"], clean_df["value"]),
                f"Corrupted_TS_{i}": (corr_df["time"], corr_df["value"])
                }

            plot_time_series_comparison(
                time_series_dict,
                title=f"{config.SYN_EXPORT_TITLE}_Comparison_{i}",
                output_dir=config.TO_COMPARISON_PLOTS_DIR
            )
            del time_series_dict
    
    print("Triggering Aggregation Pipeline")
    aggregation_pipeline(is_demo_execution, restore)

    # plots restored datasets to the original ones, for visual confirmation
    if plot:
        for i in range(config.AMOUNT_OF_INDIVIDUAL_SERIES):

            clean_df = import_dataframe_from_csv(f"{config.SYN_EXPORT_DATA_NAME}_{i}_clean")

            for method in config.INTERPOLATION_METHODS:
                
                aggregated_df = import_dataframe_from_csv(
                    filename = f"{config.SYN_EXPORT_DATA_NAME}_{i}_{method}", 
                    input_dir = traverse_to_method_dir(config.TO_AGGREGATED_DATA_DIR, method))
                
                time_series_dict = {
                    f"Clean_TS{i}": (clean_df["time"], clean_df["value"]),
                    f"{method}_TS{i}": (aggregated_df["time"], aggregated_df["value"])
                    }

                plot_time_series_comparison(
                    time_series_dict, 
                    title=f"{method}-Interpolation_Comparison_TS{i}",
                    output_dir=traverse_to_method_dir(config.TO_INTERPOLATION_PLOTS_DIR, method)
                    )
                del time_series_dict, aggregated_df

    print("Triggering Clustering Pipeline")
    clustering_pipeline(comp_dist=compute_dist, normalize=normalize)

    
    

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

    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Apply Z-normalization to each time series segment before computing or using distances."
    )

    args = parser.parse_args()

    if args.mode != "demo" and args.new_data:
        parser.error("The --new_data flag is only allowed when --mode is set to demo.")

    if args.mode == "demo":
        run_prototype(
            generate_data=args.new_data, 
            restore=args.restore,
            compute_dist=args.dist,
            normalize=args.normalized,
            plot=args.comp_img
        )
    elif args.mode == "prod":
        run_final()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()