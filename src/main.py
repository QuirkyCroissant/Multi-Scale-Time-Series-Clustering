import argparse
from data_evaluation import initialise_specific_evaluation
from data_generation import create_multiple_time_series, convert_time_series_to_dataframe
from data_corruption import corrupt_time_series_data
from data_restoration import restore_time_series_data
from data_clustering import start_clustering_pipeline
from graph_analysis import initiate_graph_analysis
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
    matrix is used for the subsequent clustering. Returns which aggregation method was used,
    on which clustering was used.'''
    start_clustering_pipeline(comp_dist, normalize)
    
def graph_clustering_pipeline(aggregation_method=config.DEFAULT_INTERPOLATION_METHOD,
                              comp_dist=False):
    initiate_graph_analysis(aggregation_method=aggregation_method, compute_dist=comp_dist)
    

def run_prototype(generate_data, 
                  restore=False, 
                  compute_dist=False,
                  normalize=False,
                  plot=False,
                  cluster_methodology=config.DEFAULT_CLUSTERING_METHOD
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
                    input_dir = traverse_to_method_dir(config.TO_AGGREGATED_DATA_DIR, method)
                    )
                
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

    if compute_dist and cluster_methodology is None:
        print("Only computing distance matrix, no clustering requested.")
        start_clustering_pipeline(compute_dist=True, normalize=normalize, stop_clustering=True)

    elif cluster_methodology in config.CLUSTERING_METHODS:
        print("Triggering Clustering Pipeline")
        clustering_pipeline(compute_dist=compute_dist, normalize=normalize)
    elif cluster_methodology in config.GRAPH_CLUSTERING_METHODS:
        print("Triggering Graph Analysis Pipeline")
        graph_clustering_pipeline(aggregation_method=config.DEFAULT_INTERPOLATION_METHOD ,compute_dist=compute_dist)
    else:
        print("No clustering method provided")
    

def run_final():
    '''TODO: production ready implementation which will be 
       implemented after an sufficient prototyp'''
    print("Running Application in Production mode:")
    pass



def run_evaluation(mode: str, metrics=[]):
    '''Executes Evaluation modeus that looks at the already existing log files and evaluates the results'''
    print("Running Application in Evaluation mode:")
    print(f"Running Evaluation: {mode}")
    print(f"Using metrics: {metrics}")
    initialise_specific_evaluation(mode, metrics)

def override_config_with_args(args):
    
    if getattr(args, "gen_amount", None) is not None:
        config.AMOUNT_OF_INDIVIDUAL_SERIES = args.gen_amount

    if getattr(args, "restore_method", None) is not None:
        config.DEFAULT_INTERPOLATION_METHOD = args.restore_method

    if getattr(args, "dist_method", None) is not None:
        config.DEFAULT_DISSIMILARITY = args.dist_method
    if getattr(args, "dist_radius", None) is not None:
        config.FASTDTW_RADIUS = args.dist_radius

    if getattr(args, "cluster_method", None) is not None:
        config.DEFAULT_CLUSTERING_METHOD = args.cluster_method
    if getattr(args, "cluster_k", None) is not None:
        config.DEFAULT_AMOUNT_OF_CLUSTERS = args.cluster_k


def main():

    parser = argparse.ArgumentParser(
        description="Main Script for Time Series Clustering"
    )

    # ----------------
    # Global Config Overwrites -> put into a parent parser
    # ----------------

    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("--gen_amount", type=int, help="Override amount of synthetic time series to generate")
    global_parser.add_argument("--restore_method", type=str, choices=config.INTERPOLATION_METHODS, help="Override interpolation method for restoration")
    global_parser.add_argument("--dist_method", type=str, choices=config.DISSIMILARITY_MEASURES, help="Override dissimilarity method (fastDTW, dtw, pearson)")
    global_parser.add_argument("--dist_radius", type=int, help="Override radius used for fastDTW")
    global_parser.add_argument("--cluster_method", type=str, choices=config.CLUSTERING_METHODS + config.GRAPH_CLUSTERING_METHODS, help="Override clustering method")
    global_parser.add_argument("--cluster_k", type=int, help="Override k amount for clustering")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ----------------
    # Demo Mode
    # ----------------

    parser_demo = subparsers.add_parser("demo", parents=[global_parser], help="Synthetic dataset generation and testing")

    parser_demo.add_argument(
        "--new_data",
        action="store_true",
        help="Include this(only in demo mode) to generate corrupted, synthetic data."
    )
    
    parser_demo.add_argument(
        "--comp_img",
        action="store_true",
        help="Save comparison plot of synthetic time series versions (experiments/plots)"
    )

    parser_demo.add_argument(
        "--restore",
        action="store_true",
        help="Aggregate, interpolate and save faulty input data (data/restored)"
    )

    parser_demo.add_argument(
        "--distance",
        action="store_true",
        help="Compute and save the (dis)-/similarity measures (experiments/distance_matrices)"
    )

    parser_demo.add_argument(
        "--normalized",
        action="store_true",
        help="Apply Z-normalization to each time series before clustering"
    )

    # ----------------
    # Prod Mode
    # ----------------

    parser_prod = subparsers.add_parser("prod", parents=[global_parser], help="Run production mode for real datasets")
    
    # TODO: TBD same as demo aggregation, dist and clustering arguments

    # ----------------
    # Eval Mode
    # ----------------
    parser_eval = subparsers.add_parser("eval", help="Analyze prior results and clustering logs")
    eval_subparsers = parser_eval.add_subparsers(dest="eval_mode", required=True)

    # Restoration Demo Eval
    parser_eval_agg = eval_subparsers.add_parser(
        "aggregation", 
        help="Evaluate interpolation/restoration results"
    )
    parser_eval_agg.add_argument(
        "--metrics", nargs='+', 
        required=True,
        choices=config.INTERPOLATION_METRICS,
        help="Metrics to compute(MSE, MAPE)"
    )

    # Distance/Clustering Demo Eval
    parser_eval_cluster_demo = eval_subparsers.add_parser(
        "clustering_demo", 
        help="Evaluate synthetic clustering results"
    )
    parser_eval_cluster_demo.add_argument(
        "--metrics", nargs='+', 
        required=True, 
        choices=config.CLUSTERING_EXTERNAL_METRICS,
        help="Metrics to compute (ARI, (A)-/RAND, NMI etc)"
    )

    # Clustering Prod
    # TODO: Check for approval

    #parser_eval_cluster_prod = eval_subparsers.add_parser(
    #    "clustering_prod", 
    #    help="Evaluate production clustering results"
    #)
    #parser_eval_cluster_prod.add_argument(
    #    "--metrics", nargs='+', 
    #    required=True, 
    #    choices=config.CLUSTERING_INTERNAL_METRICS,
    #    help="Metrics to compute (Silhouette, Modularity etc)"
    #)

    # ----------------
    # Handle Arguments
    # ----------------

    args = parser.parse_args()

    if args.dist_radius is not None and args.dist_method != "fastDTW":
        parser.error("--dist_radius can only be used if --dist_method is 'fastDTW'")
    
    if args.cluster_k is not None:
        if args.cluster_method is None or args.cluster_method not in ['kmedoids', "hierachical"]:
            parser.error("--cluster_k can only be used if --cluster_method is 'kmedoids' or 'hierachical'")

    override_config_with_args(args)

    if args.mode == "demo":
        run_prototype(
            generate_data=args.new_data, 
            restore=args.restore,
            compute_dist=args.distance,
            normalize=args.normalized,
            plot=args.comp_img,
            cluster_methodology=args.cluster_method 
        )
    elif args.mode == "prod":
        run_final()

    elif args.mode == "eval":
        
        if args.eval_mode == "aggregation":
            run_evaluation(mode=args.eval_mode, metrics=args.metrics)
        elif args.eval_mode == "clustering_demo":
            run_evaluation(mode=args.eval_mode, metrics=args.metrics)
        elif args.eval_mode == "clustering_prod":
            run_evaluation(mode=args.eval_mode, metrics=args.metrics)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()