import config
from project_utilities import (import_dataframe_from_csv_indexed, 
                               export_distance_matrix, 
                               import_distance_matrix,
                               plot_silhouette_score,
                               plot_kmedoid_results,
                               plot_hierachical_results,
                               import_restored_data_as_numpy,
                               traverse_to_method_dir)
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


def convert_to_segmented_series(data, window_length=config.SEGMENTATION_WINDOW):
    '''Method takes in time series data in form of a dataframe and converts it into a 2d numpyarray
    and returns a segmented 2d numpy array by a segmented window length defined by the passed 
    argument. The if there is not enough data to be fully segmented in the series, the data points 
    that do not fully reach a new segmentation group will get dropped. 
    
    eg.: if we have 100 entries and a segment length of 24, than we will have 4 total 
    segments, which equals to 24*4= 96 segmented data pints, last 4 get dropped.'''
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
        return i,j, dist(sequences[i], sequences[j])
    
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
                                    is_normalized=False):
    '''Starts clustering procedure on a retrieved distance matrix applying either `kmedoid` or `hierarchical`'''
    if cluster_method == "kmedoids":
        if k is None:
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
            print(f"\nOptimal k found: {best_k}")
            k = best_k

            if save_plots:
                plot_silhouette_score(k_values, silhoutte_scores, is_normalized)

        model = KMedoids(n_clusters=k, 
                         metric='precomputed', 
                         random_state=config.RANDOM_SEED)
        cluster_labels = model.fit_predict(distance_matrix)
        return cluster_labels, model
    
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
        return cluster_labels, model
        
    else:
        raise ValueError(f"Unsupported clustering method. {cluster_method}")
           
    

def start_clustering_pipeline(compute_dist=False, 
                              normalize=False,
                              aggregation_method=config.DEFAULT_INTERPOLATION_METHOD) -> str:
    '''Starts the whole clustering process, passing aggregated data through a segmentation preprocessing
    function, computing and saving the associated dissimilarity matrix and later cluster according to 
    the given distance matrix, also able to differenciate between data normalization or not.'''
    if compute_dist:
        
        series_matrix: np.ndarray = import_restored_data_as_numpy(traverse_to_method_dir(
            config.TO_AGGREGATED_DATA_DIR, 
            aggregation_method
        ))
        
        #sequences = convert_to_segmented_series(time_series_data, config.SEGMENTATION_WINDOW)
        if normalize:
            normalized_series_matrix = np.apply_along_axis(zscore, 1, series_matrix)
            series_matrix = normalized_series_matrix
           
        distance_matrix = compute_distance_matrix(series_matrix, 
                                                  method=config.DEFAULT_DISSIMILARITY,
                                                  normalize=normalize
                                                  )
        
        print("Distance matrix statistics:")
        print(distance_matrix.shape)
        print("first top left values")
        print(distance_matrix[:5, :5])
        print("minimum, maximum and average value")
        print(np.min(distance_matrix), np.max(distance_matrix), np.mean(distance_matrix))
        print("normalized" if normalize else "not normalized")
        
        export_distance_matrix(distance_matrix, 
                               method=config.DEFAULT_DISSIMILARITY,
                               normalized=normalize
                               )


        print("Completed Dissimilarity Matrix Computation.")
    
    
    labels, model = initiate_clustering_computation(
        import_distance_matrix(
            filename=config.SYN_EXPORT_DIST_MATRIX_NAME, 
            method=config.DEFAULT_DISSIMILARITY,
            date=None,
            is_normalize=normalize),
            cluster_method=config.DEFAULT_CLUSTERING_METHOD,
            k=config.DEFAULT_AMOUNT_OF_CLUSTERS,
            is_normalized=normalize
    )

    print("Cluster assignments:", labels)

    
    series_matrix: np.ndarray = import_restored_data_as_numpy(traverse_to_method_dir(
            config.TO_AGGREGATED_DATA_DIR, 
            aggregation_method
        ))
        
    if normalize:
            normalized_series_matrix = np.apply_along_axis(zscore, 1, series_matrix)
            series_matrix = normalized_series_matrix

    if config.DEFAULT_CLUSTERING_METHOD == "kmedoids":
        plot_kmedoid_results(series_matrix,
                            labels, 
                            model,
                            normalize)
    elif config.DEFAULT_CLUSTERING_METHOD == "hierarchical":
        plot_hierachical_results(series_matrix,
                            labels,
                            normalize,
                            method="average",
                            k=config.DEFAULT_AMOUNT_OF_CLUSTERS,
                            dissimilarity_matrix=import_distance_matrix(
                                                    filename=config.SYN_EXPORT_DIST_MATRIX_NAME, 
                                                    method=config.DEFAULT_DISSIMILARITY,
                                                    date=None,
                                                    is_normalize=normalize
                                                    )
        )
    else:
        raise ValueError(f"Unsupported clustering method. Supported methods are: {config.CLUSTERING_METHODS}")
    
    return aggregation_method
    