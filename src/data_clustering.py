import config
from project_utilities import (import_dataframe_from_csv_indexed, 
                               export_distance_matrix, 
                               import_distance_matrix,
                               plot_silhouette_score,
                               plot_kmedoid_results)
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
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
                            method=config.DEFAULT_DISSIMILARITY):
    '''
    Computes the pair-wise distance of the passed segmented sequence by using Fast-Dynamic Time
    Warping(DTW or fastDTW) dissimilarity measure. It utilizises numerous performant methodologies to realise it, 
    like only computing the upper triangular matrix(pairs variable) of the symmetric distance matrix and
    it leverages multithreading functionality provided by the joblib library.
    '''

    N = len(sequences)
    distance_matrix = np.zeros((N, N))
    
    if method == "fastDTW":
        distance_func = lambda x, y: fastdtw(x, y)[0]
    elif method == "dtw":
        distance_func = lambda x, y: dtw(x, y).distance
    else:
        raise ValueError(f"Unsupported dissimilarity measure. {method}")
        

    def compute_distance_pair(i, j, dist):
        return i,j, dist(sequences[i], sequences[j])
    
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    results = Parallel(n_jobs=-1)(
        delayed(compute_distance_pair)(i, j, distance_func) for i, j in tqdm(pairs, 
                                                              desc=f"Computing {method} distances.")
    )

    for i, j, dist in results:
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist
    
    return distance_matrix
    

def initiate_clustering_computation(distance_matrix: np.ndarray, 
                                    cluster_method=config.DEFAULT_CLUSTERING_METHOD,
                                    k=None,
                                    max_k=config.K_MEDOIDS_DEFAULT_MAX_CLUSTERING_AMOUNT,
                                    save_plots=True):
    '''Starts clustering procedure on a retrieved distance matrix applying either `kmedoid` or `hierarchical`'''
    if cluster_method == "kmedoids":
        if k is None:
            print("No k provided. Optimizing k using silhoutte score...")
            silhoutte_scores = []
            k_values = range(2, max_k + 1)
            
            for curr_k in tqdm(k_values, desc="Optimizing k"):
                model = KMedoids(n_clusters=curr_k, metric='precomputed', random_state=config.RANDOM_SEED)
                labels = model.fit_predict(distance_matrix)
                silhoutte_scores.append(silhouette_score(distance_matrix, labels, metric='precomputed'))
                print(f"k={curr_k}: Silhoutte Score: {silhoutte_scores[-1]:.3f}")
                
            best_k = k_values[silhoutte_scores.index(max(silhoutte_scores))]
            print(f"\nOptimal k found: {best_k}")
            k = best_k

            if save_plots:
                plot_silhouette_score(k_values, silhoutte_scores)

        model = KMedoids(n_clusters=k, 
                         metric='precomputed', 
                         random_state=config.RANDOM_SEED)
        cluster_labels = model.fit_predict(distance_matrix)
        return cluster_labels, model
    
    elif cluster_method == "hierarchical":
        # TODO: implement hierarchical clustering
        pass
    else:
        raise ValueError(f"Unsupported clustering method. {cluster_method}")
           
    

def start_clustering_pipeline(compute_dist=False, 
                                aggregation_method=config.DEFAULT_INTERPOLATION_METHOD):
    '''Starts the whole clustering process, passing aggregated data through a segmentation preprocessing
    function, computing and saving the associated dissimilarity matrix and later cluster according to 
    the given distance matrix'''
    if compute_dist:
        time_series_data: pd.DataFrame = import_dataframe_from_csv_indexed(
            config.SYN_EXPORT_DATA_NAME + '_' + config.DEFAULT_INTERPOLATION_METHOD, 
            restored=True)
        
        sequences = convert_to_segmented_series(time_series_data, config.SEGMENTATION_WINDOW)
        distance_matrix = compute_distance_matrix(sequences, method=config.DEFAULT_DISSIMILARITY)
        
        print("Distance matrix statistics:")
        print(distance_matrix.shape)
        print("first top left values")
        print(distance_matrix[:5, :5])
        print("minimum, maximum and average value")
        print(np.min(distance_matrix), np.max(distance_matrix), np.mean(distance_matrix))

        export_distance_matrix(distance_matrix, method=config.DEFAULT_DISSIMILARITY)



        print("Completed Dissimilarity Matrix Computation.")

    # TODO: Passing computed matrix into clustering logic
    labels, model = initiate_clustering_computation(import_distance_matrix(filename=config.SYN_EXPORT_DIST_MATRIX_NAME, 
                                 method=config.DEFAULT_DISSIMILARITY,
                                 date=None),
                                 cluster_method=config.DEFAULT_CLUSTERING_METHOD,
                                 k=None
    )

    #TODO visualise the labeled sequences from kmedoid
    
    time_series_data: pd.DataFrame = import_dataframe_from_csv_indexed(
            config.SYN_EXPORT_DATA_NAME + '_' + config.DEFAULT_INTERPOLATION_METHOD, 
            restored=True)
        
    segmented_seq = convert_to_segmented_series(time_series_data, config.SEGMENTATION_WINDOW)
    plot_kmedoid_results(time_series_data, segmented_seq,labels, model)
    