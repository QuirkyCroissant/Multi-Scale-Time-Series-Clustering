import numpy as np

# defines how many time series should be generated for the synthetic data sets
AMOUNT_OF_INDIVIDUAL_SERIES = 10

# Preset meta configurations for synthetic time series profiles
SYN_SERIES_PROFILES = {
    "low_flat": {
        "baseline": 8,
        "daily_amp": 0.5,
        "weekly_amp": 0.0,
        "holiday_amp": 0.0,
        "slope": 0.0,
        "noise_level": 1.0,
    },
    "regular": {
        "baseline": 10,
        "daily_amp": 1.5,
        "weekly_amp": 1.0,
        "holiday_amp": 0.0,
        "slope": 0.0,
        "noise_level": 1.5,
    },
    "vacation_heavy": {
        "baseline": 10,
        "daily_amp": 1.2,
        "weekly_amp": 0.5,
        "holiday_amp": 2.0,
        "slope": 0.0,
        "noise_level": 1.2,
    },
    "growth": {
        "baseline": 10,
        "daily_amp": 1.0,
        "weekly_amp": 0.5,
        "holiday_amp": 0.0,
        "slope": 0.05,
        "noise_level": 1.0,
    }
}

PERIOD_LENGTH = 365


# (POTENTIALLY OBSOLETE) Constants that are controlling the time series data generation for the prototype
SAMPLES_PER_DAY = 24


SYN_EXPORT_TITLE = "Synthetic_Time_Series"
SYN_EXPORT_DATA_NAME = "ts_demo_data"
RANDOM_SEED = 69


#### corruption parameters

CORRUPTION_PROBS = {
    "intact": 0.40,       # chance to leave day intact
    "partial": 0.30,      # chance to randomly delete some hours
    "reduce": 0.30        # chance to fully reduce to a single measurement
}


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
DEFAULT_DISSIMILARITY = "dtw"
DISSIMILARITY_MEASURES = ["fastDTW", "dtw"]

### clustering parameters
DEFAULT_CLUSTERING_METHOD = "kmedoids"
K_MEDOIDS_DEFAULT_CLUSTER_AMOUNT = 3
K_MEDOIDS_DEFAULT_MAX_CLUSTERING_AMOUNT = 10

