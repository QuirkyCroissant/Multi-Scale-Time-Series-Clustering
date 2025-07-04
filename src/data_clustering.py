import config
from project_utilities import (get_restored_prod_series_dir, 
                               export_distance_matrix, 
                               import_distance_matrix,
                               plot_silhouette_score,
                               plot_kmedoid_results,
                               plot_hierachical_results,
                               import_restored_data_as_numpy,
                               traverse_to_method_dir,
                               prepare_default_clustering_log, 
                               export_clustering_log,
                               get_cophenetic_corr)
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from scipy.stats import zscore, pearsonr
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from fastdtw import fastdtw
from dtw import dtw
import time


def convert_to_segmented_series(data, window_length=config.SEGMENTATION_WINDOW):
    '''
    Deprecated since prototype-v1
    
    Method takes in time series data in form of a dataframe and converts it into a 2d numpyarray
    and returns a segmented 2d numpy array by a segmented window length defined by the passed 
    argument. The if there is not enough data to be fully segmented in the series, the data points 
    that do not fully reach a new segmentation group will get dropped. 
    
    eg.: if we have 100 entries and a segment length of 24, than we will have 4 total 
    segments, which equals to 24*4= 96 segmented data pints, last 4 get dropped.
    '''
    data = np.asarray(data)
    total_segmentations = len(data) // window_length
    return data[:total_segmentations * window_length].reshape(-1, window_length)


def compute_distance_matrix(sequences: np.ndarray, 
                            method=config.DEFAULT_DISSIMILARITY,
                            normalize=False):
    '''
    Computes the pair-wise (dis)similarity matrix for a set of time series. Supports Fast-Dynamic Time
    Warping(DTW or fastDTW) dissimilarity and Pearson Correlation as similarity measures. 
    It utilizises numerous performant methodologies to realise it, like only computing the upper 
    triangular matrix(pairs variable) of the symmetric distance matrix and it leverages multithreading 
    functionality provided by the joblib library.
    '''

    N = len(sequences)
    distance_matrix = np.zeros((N, N))

    if normalize:
        tqdm_desc = f"Computing {method} distances with normalization"
    else:
        tqdm_desc = f"Computing {method} distances without normalization"
    
    if method == "fastDTW":
        distance_func = lambda x, y: fastdtw(x, y, radius=config.FASTDTW_RADIUS)[0]
    elif method == "dtw":
        distance_func = lambda x, y: dtw(x, y).distance
    elif method == "pearson":
        distance_func = lambda x, y: max(pearsonr(x, y)[0] , 0)
    else:
        raise ValueError(f"Unsupported dissimilarity measure. {method}")
        

    def compute_distance_pair(i, j, dist):
        try:
            return i,j, dist(sequences[i], sequences[j])
        except ValueError as e:
            print(f"WARN: Failed {method} between series {i} and {j}. Error: {e}")
            return i,j, np.inf
    
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    results = Parallel(n_jobs=-1)(
        delayed(compute_distance_pair)(i, j, distance_func) for i, j in tqdm(pairs, 
                                                              desc=tqdm_desc)
    )

    for i, j, dist in results:
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist
    
    return distance_matrix
    

def initiate_clustering_computation(distance_matrix: np.ndarray, 
                                    cluster_method=config.DEFAULT_CLUSTERING_METHOD,
                                    k=None,
                                    max_k=config.K_MEDOIDS_DEFAULT_MAX_CLUSTERING_AMOUNT,
                                    save_plots=True,
                                    is_normalized=False,
                                    is_prod=False):
    '''Starts clustering procedure on a retrieved distance matrix applying either `kmedoid` or `hierarchical`'''
    start = time.time()
    best_silhoutte_score = 0
    
    if cluster_method == "kmedoids":
        if k is None or config.OPTIMIZE_CLUSTER_K:
            print("No k provided. Optimizing k using silhoutte score...")
            silhoutte_scores = []
            max_allowed_k = min(max_k, distance_matrix.shape[0] - 1)
            k_values = range(2, max_allowed_k + 1)
            
            for curr_k in tqdm(k_values, desc="Optimizing k"):
                model = KMedoids(n_clusters=curr_k, metric='precomputed', random_state=config.RANDOM_SEED)
                labels = model.fit_predict(distance_matrix)
                silhoutte_scores.append(silhouette_score(distance_matrix, labels, metric='precomputed'))
                print(f"k={curr_k}: Silhoutte Score: {silhoutte_scores[-1]:.3f}")
                
            best_k = k_values[silhoutte_scores.index(max(silhoutte_scores))]
            best_silhoutte_score = max(silhoutte_scores)
            print(f"\nOptimal k found: {best_k}")
            k = best_k

            if save_plots:
                plot_silhouette_score(k_values, silhoutte_scores, is_normalized)

        model = KMedoids(n_clusters=k, 
                         metric='precomputed', 
                         random_state=config.RANDOM_SEED)
        cluster_labels = model.fit_predict(distance_matrix)

        time_diff = time.time() - start
        log = prepare_default_clustering_log(clustering_method=cluster_method,
                                             distance_measure=config.DEFAULT_DISSIMILARITY,
                                             normalized=is_normalized,
                                             n_clusters=k,
                                             labels=cluster_labels,
                                             cluster_sizes=[list(cluster_labels).count(i) for i in range(k)],
                                             silhouette_score=best_silhoutte_score if best_silhoutte_score > 0 else None,
                                             radius=config.FASTDTW_RADIUS,
                                             medoid_indices=model.medoid_indices_.tolist(),
                                             computational_time=time_diff,
                                             random_seed=config.RANDOM_SEED)
        
        export_clustering_log(log, is_prod=is_prod)

        return cluster_labels, model, k
    
    elif cluster_method == "hierarchical":
        
        print("Running Agglomerative Hierarchical Clustering with average linkage")

        if k is not None:
            print(f"Clustering using fixed number of clusters (k={k})")
            model = AgglomerativeClustering(
                metric='precomputed',
                linkage='average',
                n_clusters=k
            )
            cluster_labels = model.fit_predict(distance_matrix)
        else:
            print(f"No k provided: running fully unsupervisied clustering (distance threshold = 0)")
            model = AgglomerativeClustering(
                metric='precomputed',
                linkage='average',
                distance_threshold=0,
                n_clusters=None
            )
            cluster_labels = model.fit_predict(distance_matrix)


        time_diff = time.time() - start
        log = prepare_default_clustering_log(clustering_method=cluster_method,
                                            distance_measure=config.DEFAULT_DISSIMILARITY,
                                            normalized=is_normalized,
                                            n_clusters=k,
                                            labels=cluster_labels,
                                            cluster_sizes=[list(cluster_labels).count(i) for i in range(k)] if k is not None else None,
                                            silhouette_score=best_silhoutte_score,
                                            cophenetic_corr=get_cophenetic_corr(distance_matrix, "average"),
                                            radius=config.FASTDTW_RADIUS,
                                            computational_time=time_diff,
                                            random_seed=config.RANDOM_SEED)
        
        export_clustering_log(log, is_prod=is_prod)

        return cluster_labels, model, k
        
    else:
        raise ValueError(f"Unsupported clustering method. {cluster_method}")
           
    

def start_clustering_pipeline(compute_dist=False, 
                              normalize=False,
                              aggregation_method=config.DEFAULT_INTERPOLATION_METHOD,
                              stop_clustering=False,
                              is_prod=False):
    '''Starts the whole clustering process, passing aggregated data through a segmentation preprocessing
    function, computing and saving the associated dissimilarity matrix and later cluster according to 
    the given distance matrix, also able to differenciate between data normalization or not.
    If the user only wants to calculate the distance measures than the clustering process can be aborded 
    preemptively with the stop_clustering parameter.'''

    if is_prod:
        clustering_source_data_path = get_restored_prod_series_dir(aggregation_method)
    else:
        clustering_source_data_path = traverse_to_method_dir(
            config.TO_AGGREGATED_DATA_DIR, 
            aggregation_method
        )
    

    if compute_dist:
        
        series_matrix: np.ndarray = import_restored_data_as_numpy(clustering_source_data_path, is_prod=is_prod)
        
        #sequences = convert_to_segmented_series(time_series_data, config.SEGMENTATION_WINDOW)
        if normalize:
            normalized_series_matrix = np.apply_along_axis(zscore, 1, series_matrix)
            series_matrix = normalized_series_matrix
           
        distance_matrix = compute_distance_matrix(series_matrix, 
                                                  method=config.DEFAULT_DISSIMILARITY,
                                                  normalize=normalize
                                                  )
        
        # Checking for NaNs and np.inf from invalid dtw or pearson runs and converting them
        finite_indication_boolean_matrix = np.isfinite(distance_matrix)
        if not np.all(finite_indication_boolean_matrix):
            max_series_value = np.max(distance_matrix[finite_indication_boolean_matrix])
            fallback_value = max_series_value + 1
            distance_matrix[~finite_indication_boolean_matrix] = fallback_value
            print(f"WARN: Found NaN or Inf values in distance matrix. Replaced with fallback value: {fallback_value}")
        
        print("Distance matrix statistics:")
        print(distance_matrix.shape)
        print("first top left values")
        print(distance_matrix[:5, :5])
        print("minimum, maximum and average value")
        print(np.min(distance_matrix), np.max(distance_matrix), np.mean(distance_matrix))
        print("normalized" if normalize else "not normalized")
        
        export_distance_matrix(distance_matrix, 
                               method=config.DEFAULT_DISSIMILARITY,
                               normalized=normalize,
                               aggregation_method=aggregation_method,
                               is_prod=is_prod
                               )


        print("Completed Dissimilarity Matrix Computation.")
        if stop_clustering:
            print("Clustering process preemptively aborted. Only Distance calculated")
            return
    

    labels, model, k = initiate_clustering_computation(
        import_distance_matrix(
            filename=config.SYN_EXPORT_DIST_MATRIX_NAME, 
            method=config.DEFAULT_DISSIMILARITY,
            is_normalize=normalize,
            aggregation_method=aggregation_method,
            date=None,
            is_prod=is_prod
            ),
            cluster_method=config.DEFAULT_CLUSTERING_METHOD,
            k=config.DEFAULT_AMOUNT_OF_CLUSTERS if not config.OPTIMIZE_CLUSTER_K else None,
            is_normalized=normalize,
            is_prod=is_prod
    )

    print("Cluster assignments:", labels)

    
    series_matrix: np.ndarray = import_restored_data_as_numpy(clustering_source_data_path, is_prod=is_prod)
        
    if normalize:
            normalized_series_matrix = np.apply_along_axis(zscore, 1, series_matrix)
            series_matrix = normalized_series_matrix

    if config.DEFAULT_CLUSTERING_METHOD == "kmedoids":
        plot_kmedoid_results(series_matrix,
                            labels, 
                            model,
                            normalize,
                            is_prod=is_prod)
    elif config.DEFAULT_CLUSTERING_METHOD == "hierarchical":
        plot_hierachical_results(series_matrix,
                            labels,
                            normalize,
                            method="average",
                            k=k,
                            dissimilarity_matrix=import_distance_matrix(
                                                    filename=config.SYN_EXPORT_DIST_MATRIX_NAME, 
                                                    method=config.DEFAULT_DISSIMILARITY,
                                                    is_normalize=normalize,
                                                    aggregation_method=aggregation_method,
                                                    date=None,
                                                    is_prod=is_prod
                                                    ),
                            is_prod=is_prod
        )
    else:
        raise ValueError(f"Unsupported clustering method. Supported methods are: {config.CLUSTERING_METHODS}")
    
    