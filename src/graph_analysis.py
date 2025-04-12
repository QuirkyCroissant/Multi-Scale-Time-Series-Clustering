import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

import config
from data_clustering import compute_distance_matrix
from project_utilities import (import_restored_data_as_numpy,
                               traverse_to_method_dir,
                               export_distance_matrix,
                               import_distance_matrix)


def transform_series_into_network(series_matrix: np.ndarray, 
                                  compute_dist=True,
                                  threshold: float = None,
                                  ):
    
    print("Starting Transformation Process of Time Series into Network Graph...")
    N = series_matrix.shape[0]
    G = nx.Graph()
    
    for i in range(N):
        G.add_node(i, label=f"TS_{i}")

    if compute_dist:
        # pearson correlation is already normalizing the data
        pearson_dissim_matrix = compute_distance_matrix(series_matrix, 
                                                        method="pearson", 
                                                        normalize=False)
        
        print("Distance matrix statistics:")
        print(pearson_dissim_matrix.shape)
        print("first top left values")
        print(pearson_dissim_matrix[:5, :5])
        print("minimum, maximum and average value")
        print(np.min(pearson_dissim_matrix), np.max(pearson_dissim_matrix), np.mean(pearson_dissim_matrix))
        
        export_distance_matrix(pearson_dissim_matrix, 
                               method=config.DEFAULT_GRAPH_DISSIMILARITY
                               )

    else:
        pearson_dissim_matrix = import_distance_matrix(
            filename=config.SYN_EXPORT_DIST_MATRIX_NAME, 
            method=config.DEFAULT_GRAPH_DISSIMILARITY,
            date=None)
    
    for i in range(N):
        for j in range(i+1, N):
            weight = pearson_dissim_matrix[i][j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)
    
    print("Completed Transformation Process of Time Series into Network Graph.")
    return G
    
        
def visualize_graph(G: nx.Graph, labels=None):
    
    pos = nx.kamada_kawai_layout(G)
    edge_weights =[G[u][v]['weight'] for u,v in G.edges()]
    

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, 
            with_labels=True, 
            node_color="#8B0000", 
            font_color="white",
            edge_color=edge_weights, 
            width=2,
            edge_cmap=plt.cm.Reds_r,
            labels=labels)
    
    plt.title("Graph of Time Series Data Correlation (Pearson Dissimilarity)")
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"graph_visualization_{date}.png"
        

    plot_path = os.path.join(
        config.TO_GRAPH_ANALYSIS_PLOT_DIR,
        filename
    )

    plt.savefig(plot_path)
    plt.close()
    print(f"Graph visualization saved to: {plot_path}")
    
    
def apply_graph_clustering():
    pass
    


def initiate_graph_analysis(aggregation_method=config.DEFAULT_INTERPOLATION_METHOD,
                            compute_dist=False,
                            ):
    '''Beginns Graph-Analysis Portion of the project. Starts with the transformation of 
    Time Series Data into Nodes and uses Pearson Correlation as Edge weight. After dropping
    edges that do not uphold a certain threshold to make the graph more sparse, the program
    proceeds to cluster the given graph.'''

    series_matrix: np.ndarray = import_restored_data_as_numpy(traverse_to_method_dir(
            config.TO_AGGREGATED_DATA_DIR, 
            aggregation_method
        ))
    
    series_network = transform_series_into_network(series_matrix=series_matrix, 
                                                   compute_dist=compute_dist, 
                                                   threshold=config.GRAPH_THRESHOLD)
    
    
    visualize_graph(series_network)

    # TODO: Implement Graph Clustering
    apply_graph_clustering()
    
    
        



