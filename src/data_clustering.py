import config
from project_utilities import import_dataframe_from_csv_indexed, export_distance_matrix
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from fastdtw import fastdtw
from dtw import dtw


def convert_to_segmented_series(data, window_length):
    '''Method takes in time series data in form of a dataframe and converts it into a 2d numpyarray
    and returns a segmented 2d numpy array by a segmented window length defined by the passed 
    argument. The if there is not enough data to be fully segmented in the series, the data points 
    that do not fully reach a new segmentation group will get dropped. 
    
    eg.: if we have 100 entries and a segment length of 24, than we will have 4 total 
    segments, which equals to 24*4= 96 segmented data pints, last 4 get dropped.'''
    data = np.asarray(data)
    total_segmentations = len(data) // window_length
    return data[:total_segmentations * window_length].reshape(-1, window_length)


def compute_distance_matrix(sequences: np.ndarray):
    '''
    Computes the pair-wise distance of the passed segmented sequence by using Fast-Dynamic Time
    Warping(fastDTW) dissimilarity measure. It utilizises numerous performant methodologies to realise it, 
    like only computing the upper triangular matrix(pairs variable) of the symmetric distance matrix,
    it leverages multithreading functionality provided by the joblib library.
    '''

    N = len(sequences)
    distance_matrix = np.zeros((N, N))
    
    def compute_distance_pair(i, j):
        #dist, _ = fastdtw(sequences[i], sequences[j])
        alignment = dtw(sequences[i], sequences[j])
        return i,j, alignment.distance
    
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    results = Parallel(n_jobs=-1)(
        delayed(compute_distance_pair)(i, j) for i, j in tqdm(pairs, 
                                                              desc="Computing FastDTW distances.")
    )

    for i, j, dist in results:
        distance_matrix[i, j] = dist
        distance_matrix[j, i] = dist
    
    return distance_matrix
    


def initiate_clustering_process():
    '''Starts the whole clustering process, passing aggregated data through a segmentation preprocessing
    function, computing and saving the associated dissimilarity matrix and later cluster according to 
    the given distance matrix'''

    time_series_data: pd.DataFrame = import_dataframe_from_csv_indexed(
        config.SYN_EXPORT_DATA_NAME + '_linear', restored=True)
    
    # FULL_TIMESERIES_DATA = time_series_data.iloc[:int(len(time_series_data) * 1)]

    sequences = convert_to_segmented_series(time_series_data, config.SEGMENTATION_WINDOW)
    dtw_matrix = compute_distance_matrix(sequences)
    
    print("Distance matrix statistics:")
    print(dtw_matrix.shape)
    print("first top left values")
    print(dtw_matrix[:5, :5])
    print("minimum, maximum and average value")
    print(np.min(dtw_matrix), np.max(dtw_matrix), np.mean(dtw_matrix))

    export_distance_matrix(dtw_matrix, method="dtw")



    print("Completed Dissimilarity Matrix Computation.")

    # TODO: Passing computed matrix into clustering logic
    