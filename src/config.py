import numpy as np
import os

# defines how many time series should be generated for the synthetic data sets
AMOUNT_OF_INDIVIDUAL_SERIES = 10

# Preset meta configurations for synthetic time series profiles
SYN_SERIES_PROFILES = {
    "low_flat": {
        "baseline": 4,
        "daily_amp": 0.5,
        "weekly_amp": 0.0,
        "holiday_amp": 0.0,
        "slope": 0.0,
        "noise_level": .8,
    },
    "regular": {
        "baseline": 15,
        "daily_amp": 1.5,
        "weekly_amp": 1.0,
        "holiday_amp": 0.0,
        "slope": 0.005,
        "noise_level": 1.5,
    },
    "vacation_heavy": {
        "baseline": 8,
        "daily_amp": 1.2,
        "weekly_amp": 0.5,
        "holiday_amp": 4.0,
        "slope": 0.01,
        "noise_level": 0.9,
    },
    "growth": {
        "baseline": 9,
        "daily_amp": 1.0,
        "weekly_amp": 0.5,
        "holiday_amp": 0.0,
        "slope": 0.05,
        "noise_level": 1.0,
    }
}

PERIOD_LENGTH = 365
TIME_SERIES_START_DATE = "2023-01-01"



# (POTENTIALLY OBSOLETE) Constants that are controlling the time series data generation for the prototype
SAMPLES_PER_DAY = 24


SYN_EXPORT_TITLE = "Synthetic_Time_Series"
SYN_EXPORT_DATA_NAME = "ts_demo_data"
RANDOM_SEED = 69


#### Traversal Paths for I/O operations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TO_CLEAN_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "generated")
TO_CORRUPTED_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "corrupted")
TO_AGGREGATED_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "restored")

TO_DISTANCES_DIR = os.path.join(SCRIPT_DIR, "..", "experiments", "distance_matrices")
TO_INTERPOLATION_LOGS_DIR = os.path.join(SCRIPT_DIR, "..", "experiments", "logs", "interpolations")
TO_CLUSTERING_LOGS_DIR = os.path.join(SCRIPT_DIR, "..", "experiments", "logs", "clustering")

TO_DEFAULT_CLUSTERING_LOGS_DIR = os.path.join(TO_CLUSTERING_LOGS_DIR, "default")
TO_GRAPH_CLUSTERING_LOGS_DIR = os.path.join(TO_CLUSTERING_LOGS_DIR, "graph")

TO_CLEAN_DATA_PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "experiments", "plots", "generated_data")
TO_CORRUPTED_DATA_PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "experiments", "plots", "corrupted_data")
TO_COMPARISON_PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "experiments", "plots", "comparisons")

TO_INTERPOLATION_PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "experiments", "plots", "interpolations")
TO_CLUSTERING_PLOTS_DIR = os.path.join(SCRIPT_DIR, "..", "experiments", "plots", "clustering")
TO_GRAPH_ANALYSIS_PLOT_DIR = os.path.join(SCRIPT_DIR, "..", "experiments", "plots", "graph_analysis")





#### corruption parameters

CORRUPTION_PROBS = {
    "intact": 0.40,       # chance to leave day intact
    "partial": 0.30,      # chance to randomly delete some hours
    "reduce": 0.30        # chance to fully reduce to a single measurement
}

DAILY_CORRUPTION_RATE = 0.15


#### restoration parameters

DEFAULT_INTERPOLATION_METHOD = "linear"
INTERPOLATION_METHODS = [
    "linear", "time", "index", 
    "values", "pad", "nearest", 
    "zero", "slinear", "quadratic", 
    "cubic", "barycentric", "polynomial", 
    "spline", "piecewise_polynomial", 
    "pchip", "akima", "cubicspline"
    ]

### dissimilarity measure parameters

# segmentation length defines the size of segmentated chunks for the whole time series, 
# in the current case it is assumed that we deal with hourly measurements and therefore
# we want to group the hourly data into chunks of days, therefore 24
SEGMENTATION_WINDOW = 24
SYN_EXPORT_DIST_MATRIX_NAME = "distance_matrix"
DEFAULT_DISSIMILARITY = "fastDTW"
DISSIMILARITY_MEASURES = ["fastDTW", "dtw", 'pearson']
FASTDTW_RADIUS = 3

### clustering parameters
DEFAULT_CLUSTERING_METHOD = "kmedoids"
CLUSTERING_METHODS = ["kmedoids", "hierarchical"]
K_MEDOIDS_DEFAULT_CLUSTER_AMOUNT = 3
K_MEDOIDS_DEFAULT_MAX_CLUSTERING_AMOUNT = 10

# if we want to compare it to kmedoids cluster evaluation and make the clustering 
# semi-unsupervised
DEFAULT_AMOUNT_OF_CLUSTERS = 4

### Graph parameters

# lets keep only edges which have a similarity of at least 30%
GRAPH_THRESHOLD = 0.3
DEFAULT_GRAPH_DISSIMILARITY = DISSIMILARITY_MEASURES[2]

DEFAULT_GRAPH_CLUSTERING_METHOD = "louvain"


### Evaluation choices

INTERPOLATION_METRICS = ['MSE', 'MAPE']
CLUSTERING_EXTERNAL_METRICS = ['ARI', 'RAND', 'ARAND', 'NMI']
CLUSTERING_INTERNAL_METRICS = ['silhouette', 'modularity']



