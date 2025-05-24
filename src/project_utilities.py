import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import networkx as nx
from typing import List, Optional, Dict, Union
from scipy.stats import zscore
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np
import config
import datetime
import os
import re
import json

def get_restored_prod_series_dir(method_name):
    '''Retrieves restored production directory for a given method'''
    return traverse_to_method_dir(os.path.join(config.TO_AGGREGATED_DATA_DIR, method_name), "prod")

def count_extracted_prod_series():
    return len([
        f for f in os.listdir(config.TO_PROD_SERIES_EXPORT_DATA_DIR) 
        if f.startswith(config.PROD_EXPORT_DATA_NAME) and f.endswith("_raw")
    ])

def import_restored_data_as_numpy(input_dir):
    '''Imports the aggregated data from a given interpolated 
    datafolder and imports it in a 2D numpy array.'''
    all_files = sorted([
        f for f in os.listdir(input_dir)
        if not f.startswith('.') 
        and os.path.isfile(os.path.join(input_dir, f))
        and "ts_prod_data" in f 
    ])
    
    data_matrix = []
    
    for file in all_files:
        current_time_series = np.loadtxt(os.path.join(input_dir,file), delimiter=';', skiprows=1, usecols=1, dtype=np.float32)
        data_matrix.append(current_time_series)
    
    return np.array(data_matrix)
    

def export_distance_matrix(np_matrix, 
                           filename=config.SYN_EXPORT_DIST_MATRIX_NAME, 
                           method=config.DEFAULT_DISSIMILARITY,
                           normalized=False,
                           aggregation_method=config.DEFAULT_INTERPOLATION_METHOD,
                           is_prod=False):
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_dir = config.TO_DISTANCES_DIR
    
    if normalized:
        filename = f"{filename}_{aggregation_method}_normalized_{method}"
    else:
        filename = f"{filename}_{aggregation_method}_raw_{method}"

    if method == "fastDTW":
        filename += f"_r{config.FASTDTW_RADIUS}"
    
    filename += f"_{date}"
    
    if is_prod:
        filepath = os.path.join(output_dir, "prod", filename)
    else:
        filepath = os.path.join(output_dir, filename)

    np.save(filepath, np_matrix)
    prod_tag = "(PROD)" if is_prod else ""
    print(f"Distance matrix{prod_tag} saved to: {filepath}")
        
        
def import_distance_matrix(filename=config.SYN_EXPORT_DIST_MATRIX_NAME,
                           method=config.DEFAULT_DISSIMILARITY,
                           is_normalize=False,
                           aggregation_method=config.DEFAULT_INTERPOLATION_METHOD,
                           date=None,
                           is_prod=False):
    '''Imports a distance matrix from the respective experiments folder and depending
    on if a specific date has not been passed it will retrieve the newest file or a
    specific one.'''
    
    radius_suffix = f"_r{config.FASTDTW_RADIUS}" if method == "fastDTW" else ""

    if date is None:
    
        if is_normalize:
            filename_without_date = f"{filename}_{aggregation_method}_normalized_{method}{radius_suffix}_"
        else:
            filename_without_date = f"{filename}_{aggregation_method}_raw_{method}{radius_suffix}_"


        dir_path = config.TO_DISTANCES_DIR
        if is_prod:
            dir_path = os.path.join(dir_path, "prod")
        
        all_files = os.listdir(dir_path)
        matching_files = []
        date_pattern = re.compile(rf"^{re.escape(filename_without_date)}(\d{{4}}-\d{{2}}-\d{{2}})\.npy$")
        
        for file in all_files:
            match = date_pattern.match(file)
            if match:
                file_date = match.group(1)
                matching_files.append((file_date, file))
        
        if not matching_files:
            error_msg = f"No distance matrix files found starting with '{filename_without_date}' and ending with '{dir_path}' ."
            raise FileNotFoundError(error_msg)
                
        matching_files.sort(key=lambda x: x[0], reverse=True)
        _, youngest_filename = matching_files[0]
        filepath = os.path.join(dir_path, youngest_filename)

    else:
        if is_normalize:
            filename_with_date = f"{filename}_{aggregation_method}_normalized_{method}{radius_suffix}_{date}.npy"
        else:
            filename_with_date = f"{filename}_{aggregation_method}_raw_{method}{radius_suffix}_{date}.npy"

        if is_prod:
            filepath = os.path.join(config.TO_DISTANCES_DIR, 
                                    "prod", 
                                    filename_with_date)
        else: 
            filepath = os.path.join(config.TO_DISTANCES_DIR, 
                                    filename_with_date)
    
    prod_tag = "(PROD)" if is_prod else ""
    print(f"Loaded distance matrix{prod_tag} from: {filepath}")

    return np.load(filepath)
    


def compute_and_save_accuracy(df, method_name, 
                              series_id=0, 
                              time_difference=0):
    '''Computes the accuracy of how similar the given dataset to the uncorrupted 
    synthetic dataset is. After computation if prints out the result and exports 
    it into the log files in the experiments folder.'''
    ts_demo_data_clean = import_dataframe_from_csv(
        filename=f"{config.SYN_EXPORT_DATA_NAME}_{series_id}_clean",
        input_dir=config.TO_CLEAN_DATA_DIR
        )
    
    values = []
    mse_value = mean_squared_error(ts_demo_data_clean['value'], df['value'])
    print(f"The Mean-Squared-Error(MSE) for using the {method_name}-method is: \n{mse_value}")
    values.append(mse_value)
    
    mape_value = mean_absolute_percentage_error(ts_demo_data_clean['value'], df['value'])
    print(f"The Mean-Absolute-Percentage-Error(MAPE) for using the {method_name}-method is: \n{mape_value}")
    values.append(mape_value)

    export_accuracy_log(values, method_name, series_id, time_difference)

    
def export_accuracy_log(values, method_name, series_id, time_difference):

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_dir = config.TO_INTERPOLATION_LOGS_DIR
    
    filename = os.path.join(output_dir, method_name, f"accuracy_log_ts_{series_id}_{method_name}_{date}.json")
    
    with open(filename, "w", encoding='utf-8') as f: 
        json.dump({"methodName": method_name, 
                   "mseScore": values[0], 
                   "mapeScore": values[-1], 
                   "executionDate": date,
                   "interpolationTime": time_difference
                   }, 
                  f, 
                  ensure_ascii=False, 
                  indent=4)

    print(f"Log file saved to: {filename}")
    
    
def prepare_default_clustering_log(clustering_method: str,
                                   distance_measure: str,
                                   normalized: bool,
                                   n_clusters: int,
                                   labels: List[int],
                                   cluster_sizes: List[int],
                                   silhouette_score: Optional[float] = None,
                                   cophenetic_corr: Optional[float] = None,
                                   radius: Optional[int] = None ,
                                   medoid_indices: Optional[List[int]] = None,
                                   computational_time: Optional[float] = None,
                                   random_seed: int = config.RANDOM_SEED,
                                   extra: Optional[Dict] = None) -> Dict:
    '''Helper function which builds a log file for the results that are being produced 
    after the clustering process with time series data.'''

    log = {
        "clusteringMethod": clustering_method,
        "distanceMeasure": distance_measure,
        "normalized": normalized,
        "nClusters": n_clusters,
        "dataQuantity": len(labels),
        "labels": labels.tolist() if isinstance(labels, np.ndarray) else labels,
        "clusterSizes": cluster_sizes.tolist() if isinstance(cluster_sizes, np.ndarray) else cluster_sizes,
        "silhouetteScore": silhouette_score,
        "copheneticCorrelation": cophenetic_corr,
        "radius": radius,
        "medoidIndices": medoid_indices.tolist() if isinstance(medoid_indices, np.ndarray) else medoid_indices,
        "timeStamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "computationalTime": computational_time,
        "randomSeed": random_seed
    }

    if extra:
        log.update(extra)
    return log

def prepare_graph_clustering_log(clustering_method: str,
                                 edge_weight_metric: str,
                                 threshold: float,
                                 normalized: bool,
                                 n_clusters: int,
                                 labels: List[int],
                                 cluster_sizes: List[int],
                                 radius: Optional[int] = None,
                                 computational_time: Optional[float] = None,
                                 random_seed: int = config.RANDOM_SEED,
                                 extra: Optional[Dict] = None) -> Dict:
    '''Helper function which builds a log file for the results that are being produced 
    after the clustering process with the graph data.'''

    log = {
        "clusteringMethod": clustering_method,
        "distanceMeasure": edge_weight_metric,
        "threshold": threshold,
        "normalized": normalized,
        "nClusters": n_clusters,
        "dataQuantity": len(labels),
        "labels": labels.tolist() if isinstance(labels, np.ndarray) else labels,
        "clusterSizes": cluster_sizes.tolist() if isinstance(cluster_sizes, np.ndarray) else cluster_sizes,
        "radius": radius,
        "timeStamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "computationalTime": computational_time,
        "randomSeed": random_seed
    }

    if extra:
        log.update(extra)
    return log


def export_clustering_log(log_data: dict, is_prod=False):
    '''Function takes in a dictionary of resulting clustering pipeline from either the default
    or the graph clustering process and saves it as a json file in the corresponding directory'''
    
    timestamp = log_data['timeStamp']

    clustering_method = log_data['clusteringMethod']
    if clustering_method == "hierarchical" and log_data['nClusters'] is not None:
        filename = f"log_{clustering_method}_capped{log_data['nClusters']}"
    else:
        filename = f"log_{clustering_method}"


    distance_measure = log_data['distanceMeasure']
    is_normalized = "n_" if log_data['normalized'] else ""
    print(log_data['nClusters'])
    is_unsupervised = "u_" if log_data['nClusters'] is None else ""
    if distance_measure == "fastDTW":
        radius = log_data['radius']
        filename += f"_{distance_measure}_r{radius}_{is_normalized}{is_unsupervised}{timestamp}.json"
    else:
        filename += f"_{distance_measure}_{is_normalized}{is_unsupervised}{timestamp}.json"

    if clustering_method in config.CLUSTERING_METHODS:
        if is_prod:
            full_path = os.path.join(config.TO_DEFAULT_CLUSTERING_LOGS_DIR, "prod", filename)
        else:
            full_path = os.path.join(config.TO_DEFAULT_CLUSTERING_LOGS_DIR, filename)
    else:
        if is_prod:
            full_path = os.path.join(config.TO_GRAPH_CLUSTERING_LOGS_DIR, "prod", filename)
        else:
            full_path = os.path.join(config.TO_GRAPH_CLUSTERING_LOGS_DIR, filename)

    with open(full_path, "w", encoding='utf-8') as f: 
        json.dump(log_data, f, ensure_ascii=False, indent=4)
    
    print(f"Clustering Log file saved to: {full_path}")


def export_dataframe_to_csv(df, 
                            filename=config.SYN_EXPORT_DATA_NAME, 
                            output_dir=None,
                            clean=False,
                            corrupted=False,
                            eval=False,
                            prod=False):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        if clean:
            output_dir = os.path.join(script_dir,"..", "data", "generated")
        elif corrupted:
            output_dir = os.path.join(script_dir,"..", "data", "corrupted")
        else:
            output_dir = os.path.join(script_dir,"..", "data")
            
    if clean:
        filename = f"{filename}_clean"
    elif corrupted:
        filename = f"{filename}_corrupted"
    
    if eval:
        filename = f"{filename}"
    
    if prod:
        filename = f"{filename}"

    filepath = os.path.join(output_dir, filename)

    df.to_csv(filepath, index=False, sep=';')
    print(f"DataFrame exported to: {filepath}")
    
def traverse_to_method_dir(traversal_string, method_name):
    '''Method that retrieves the precise subdirectory for paths that also use the interpolation methods'''
    return os.path.join(traversal_string, method_name)

def import_dataframe_from_csv(filename=config.SYN_EXPORT_DATA_NAME, input_dir=None):
    '''imports time series dataframe from csv without indexing'''
    
    if input_dir is None:
        input_dir = config.TO_CLEAN_DATA_DIR
    filepath = os.path.join(input_dir, filename)

    df = pd.read_csv(filepath, sep=';')
    return df
    
def import_dataframe_from_csv_indexed(filename=config.SYN_EXPORT_DATA_NAME, input_dir=None, restored=False):
    '''imports time series dataframe from csv and indexes the time column, 
    in order to leverage powerful and efficient DateTimeIndex functionality from pandas library'''
    
    if input_dir is None:
        if restored:
            input_dir = config.TO_AGGREGATED_DATA_DIR
        else:
            input_dir = config.TO_CLEAN_DATA_DIR
        
    filepath = os.path.join(input_dir, filename)

    df = pd.read_csv(filepath, sep=';', index_col=[0], parse_dates=[0])
    return df
    
def deindex_dataframe(dataframe):
    return dataframe.reset_index()

def plot_time_series(y, x=None, format='-', start=0, end=None,
                     title=None, xlabel=None, ylabel=None, 
                     legend=None, output_dir=None):
    plt.figure(figsize=(10, 6))

    if x is None:
        x = range(len(y))

    if isinstance(y, pd.DataFrame):
        if "time" in y.columns and "value" in y.columns:
            x = y["time"]
            y = y["value"]
        else:
            raise ValueError("DataFrame must contain 'time' and 'value' columns.")

    if isinstance(y, (list, np.ndarray, pd.Series)) and (not hasattr(y, 'ndim') or y.ndim == 1):
        plt.plot(x[start:end], y[start:end], format)
    elif isinstance(y, tuple):
        for y_i in y:
            plt.plot(x[start:end], y_i[start:end], format)
    else:
        raise ValueError("Unsupported format for y")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend:
        plt.legend(legend)

    plt.title(title)
    plt.grid(True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(script_dir, "..", "experiments", "plots", "generated_data")

    filename = f"{title}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to: {filepath}")


def plot_time_series_comparison(series_dict, title="TimeSeries_Plot",
                                output_dir=None, 
                                xlabel="Time", ylabel="Value", fmt='-', 
                                freq='D'):
    '''Plots time series and exports the image, is possible to compare multiple series.'''

    plt.figure(figsize=(10, 6))

    for label, (x, y) in series_dict.items():
        # series get normalized by frequenzy(hourly, daily etc.)
        s = pd.Series(y.values, index=pd.to_datetime(x))
        new_index = pd.date_range(start=s.index.min(), end=s.index.max(), freq=freq)
        s_reindexed = s.reindex(new_index)
        # time series now gets plotted true to its temporal scale, NaN values get ignored
        plt.plot(s_reindexed.index, s_reindexed.values, fmt, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "experiments", "plots")
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{title}_{date}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to: {filepath}")

def plot_silhouette_score(k_values, silhoutte_scores, is_normalized=False):
    '''creates silhoutte score plot and saves it into the experiments folder'''
    plt.figure()
    plt.plot(k_values, silhoutte_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters (Normalized)' 
              if is_normalized else 
              'Silhouette Score vs. Number of Clusters')
    plt.grid(True)

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    if is_normalized:
        filename = f"silhouette_score_{config.DEFAULT_DISSIMILARITY}_normalized_{date}.png"
    else:
        filename = f"silhouette_score_{config.DEFAULT_DISSIMILARITY}_raw_{date}.png"
        
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..",
                             "experiments",
                             "plots",
                             "clustering",
                             filename
                             )
    plt.savefig(plot_path)
    print(f"Silhouette score plot saved to: {plot_path}")
    plt.close()

def plot_kmedoid_results(series_matrix, 
                         labels, 
                         model, 
                         is_normalized=False,
                         is_prod=False):

    n_clusters = len(np.unique(labels))
    colors = plt.cm.get_cmap("tab10", n_clusters)
    series_length = series_matrix.shape[1]
    time_axis = np.arange(series_length)
    n_series = series_matrix.shape[0]
    

    _, axs = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

    ### Full Time Series coloured by cluster ###
    for idx, (series, label) in enumerate(zip(series_matrix,labels)):
        axs[0].plot(time_axis, series, color=colors(label), alpha=0.6, linewidth=0.8)

    axs[0].set_title(f"Full Time Series Colored by Cluster(n = {n_series})" + (" - Normalized" if is_normalized else ""))
    axs[0].set_xlabel("Time (Days)")
    axs[0].set_ylabel("Value")
    axs[0].grid(True)


    ### Cluster Medoids  ###
    for cluster_id, medoid_idx in enumerate(model.medoid_indices_):
        axs[1].plot(time_axis, series_matrix[medoid_idx], label=f"Medoid {cluster_id}", 
                    color=colors(cluster_id), linewidth=2.5)

    axs[1].set_title(f"Cluster Medoid Profiles" + (" (Normalized)" if is_normalized else ""))
    axs[1].set_xlabel("Time(Days)")
    axs[1].set_ylabel("Value")
    axs[1].legend()
    axs[1].grid(True)



    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"kmedoid_multiseries_{config.DEFAULT_DISSIMILARITY}_{'normalized' if is_normalized else 'raw'}_{date}.png"
        
    if is_prod:
        plot_path = os.path.join(
            config.TO_CLUSTERING_PLOTS_DIR,
            "prod",
            filename
        )
    else:
        plot_path = os.path.join(
            config.TO_CLUSTERING_PLOTS_DIR,
            filename
        )

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    prod_tag = "(PROD)" if is_prod else ""
    print(f"Multi-series clustering Plot{prod_tag} saved to: {plot_path}")

def get_cophenetic_corr(dist_matrix, linkage_method):
    condensed = squareform(dist_matrix)
    return cophenet(linkage(condensed, method=linkage_method), condensed)[0]

def plot_hierachical_results(series_matrix, 
                             labels, 
                             is_normalized=False, 
                             method="average", 
                             k = None, 
                             dissimilarity_matrix=None,
                             is_prod=False):
    '''Plots the results of hierarchical clustering. On the left is the dendrogram,
    and on the right is the colored time series by cluster.'''

    if dissimilarity_matrix is None:
        raise ValueError("To plot a dendrogram, you must provide a dissimilarity matrix.")
    
    n_clusters = len(np.unique(labels))
    colors = plt.cm.get_cmap("tab10", n_clusters)
    series_length = series_matrix.shape[1]
    time_axis = np.arange(series_length)
    n_series = series_matrix.shape[0]

    fig, axs = plt.subplots(1,2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 3]})

    ### Dendogram ###
    condensed = squareform(dissimilarity_matrix)
    linkage_matrix = linkage(condensed, method=method)

    # Used to cut the dendrogram if a k is defined 
    # grabs the height (distance) of the last merge before you reach k clusters
    if k:
        threshold = linkage_matrix[-(k - 1), 2]

    dendrogram(linkage_matrix, 
               labels=[f"TS_{i}" for i in range(n_series)], 
               ax=axs[0], 
               color_threshold=0)
    
    if k:
        axs[0].set_ylim(0, threshold + (0.1 * threshold))
        #axs[0].axhline(y=threshold, color='r', linestyle='--')

        
    coph_corr, _ = cophenet(linkage_matrix, condensed)
    
    fig.suptitle(f"Hierarchical Clustering Results (Cophenetic Corr: {coph_corr:.3f})", fontsize=14)
    
    dendro_title = "Dendrogram of Hierachical Clustering"
    if k:
        dendro_title += f" (k={k})"
    axs[0].set_title(dendro_title)
    axs[0].set_xlabel("Time Series")
    axs[0].set_ylabel("Distance")

    ### Full Time Series coloured by cluster ###
    for idx, (series, label) in enumerate(zip(series_matrix,labels)):
        axs[1].plot(time_axis, series, color=colors(label), alpha=0.6, linewidth=0.8)

    axs[1].set_title(f"Time Series by Cluster (n = {n_series})" + (" - Normalized" if is_normalized else ""))
    axs[1].set_xlabel("Time (Days)")
    axs[1].set_ylabel("Value")
    axs[1].grid(True)

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"hierarchical_multiseries_{config.DEFAULT_DISSIMILARITY}_"
    
    if k:
        filename += f"capped_to_{k}_"
    else:
        filename += "unsupervised_"
    
    if is_normalized:
        filename += f"normalized_{date}.png"
    else:
        filename += f"raw_{date}.png"

    if is_prod:
        plot_path = os.path.join(
            config.TO_CLUSTERING_PLOTS_DIR,
            "prod",
            filename
        )
    else:
        plot_path = os.path.join(
            config.TO_CLUSTERING_PLOTS_DIR,
            filename
        )

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    prod_tag = "(PROD)" if is_prod else ""
    print(f"Multi-series clustering Plot{prod_tag} saved to: {plot_path}")


def plot_graph_clustering_results(G: nx.Graph, 
                            series_matrix: np.ndarray, 
                            cluster_labels: list[int], 
                            method=config.DEFAULT_GRAPH_CLUSTERING_METHOD,
                            dist="pearson",
                            title_suffix: str = "",
                            is_prod=False):

    n_clusters = len(set(cluster_labels))
    colors = plt.cm.get_cmap("tab10", n_clusters)
    time_axis = np.arange(series_matrix.shape[1])

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    ### Graph Network ###
    pos = nx.kamada_kawai_layout(G)
    node_colors = [colors(label) for label in cluster_labels]

    nx.draw(G, pos,
            ax=axs[0],
            node_color=node_colors,
            with_labels=True,
            edge_color="#888",
            font_color="white",
            node_size=500,
            font_size=8)

    axs[0].set_title(f"Graph View of Time Series Clusters{title_suffix}")

    ### Full Time Series coloured by cluster ###
    for idx, (series, label) in enumerate(zip(series_matrix, cluster_labels)):
        axs[1].plot(time_axis, series, color=colors(label), alpha=0.7, linewidth=1)

    axs[1].set_title("Time Series Colored by Cluster")
    axs[1].set_xlabel("Time (Days)")
    axs[1].set_ylabel("Value")
    axs[1].grid(True)

    fig.tight_layout()
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"graph_{method}_clustering_{dist}_{date}.png"

    if is_prod:
        plot_path = os.path.join(
            config.TO_CLUSTERING_PLOTS_DIR,
            "prod",
            filename
        )
    else:
        plot_path = os.path.join(
            config.TO_CLUSTERING_PLOTS_DIR,
            filename
        )

    plt.savefig(plot_path)
    plt.close()
    prod_tag = "(PROD)" if is_prod else ""
    print(f"Graph-based clustering Visualisation{prod_tag} saved to: {plot_path}")