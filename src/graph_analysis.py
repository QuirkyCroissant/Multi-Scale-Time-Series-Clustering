import time
import networkx as nx
import community as community_louvain
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

import config
from data_clustering import compute_distance_matrix
from project_utilities import (export_clustering_log, import_restored_data_as_numpy, 
                               prepare_graph_clustering_log,
                               traverse_to_method_dir,
                               export_distance_matrix,
                               import_distance_matrix,
                               plot_graph_clustering_results)


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
            edge_cmap=plt.cm.Reds,
            labels=labels)
    
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=9)
    
    plt.title("Graph of Time Series Data Correlation (Pearson Similarity)")
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"graph_visualization_{date}.png"
        

    plot_path = os.path.join(
        config.TO_GRAPH_ANALYSIS_PLOT_DIR,
        filename
    )

    plt.savefig(plot_path)
    plt.close()
    print(f"Graph visualization saved to: {plot_path}")
    
    
def apply_graph_clustering(G: nx.Graph, method=config.DEFAULT_GRAPH_CLUSTERING_METHOD):
    '''Applies graph clustering algorithms depending on the input parameter: louvain uses the popular 
    modularity based algorithm, modularity uses a greedy heuristic and label uses label propagation'''

    start = time.time()

    if method == "louvain":
        partition = community_louvain.best_partition(G, weight='weight')
        labels = [partition[node] for node in G.nodes]
        
    elif method == "modularity":
        communities = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
        labels = [None] * len(G)
        for cid, community in enumerate(communities):
            for node in community:
                labels[node] = cid
    
    elif method == "label":
        communities = nx.algorithms.community.asyn_lpa_communities(G, weight='weight')
        for cid, community in enumerate(communities):
            for node in community:
                labels[node] = cid
    
    else:
        raise ValueError(f"Unsupported graph clustering method: {method}")
    

    return labels, time.time() - start


def initiate_graph_analysis(aggregation_method=config.DEFAULT_INTERPOLATION_METHOD,
                            clustering_method=config.DEFAULT_GRAPH_CLUSTERING_METHOD,
                            compute_dist=False,
                            is_normalized=False
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
    
    graph_cluster_labels, comp_time = apply_graph_clustering(series_network)

    series_matrix: np.ndarray = import_restored_data_as_numpy(traverse_to_method_dir(
            config.TO_AGGREGATED_DATA_DIR, 
            aggregation_method
        ))
    
    plot_graph_clustering_results(series_network, 
                                  series_matrix, 
                                  graph_cluster_labels,
                                  config.DEFAULT_GRAPH_CLUSTERING_METHOD
                                  )
    k = len(np.unique(graph_cluster_labels))
    log = prepare_graph_clustering_log(clustering_method=clustering_method,
                                            edge_weight_metric=config.DEFAULT_GRAPH_DISSIMILARITY,
                                            threshold=config.GRAPH_THRESHOLD,
                                            normalized=is_normalized,
                                            n_clusters=k,
                                            labels=graph_cluster_labels,
                                            cluster_sizes=[list(graph_cluster_labels).count(i) for i in range(k)] if k is not None else None,
                                            radius=config.FASTDTW_RADIUS,
                                            computational_time=comp_time,
                                            random_seed=config.RANDOM_SEED)
        
    export_clustering_log(log)


