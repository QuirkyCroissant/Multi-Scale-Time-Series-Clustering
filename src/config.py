import numpy as np

SYNTHETHIC_TIMESERIES_CONFIG: dict = {
  "period_length": 365,
  "baseline": 10,
  "amplitude": 40,
  "slope": 0.05,
  "noise_level": 5
}

# Constants that are controlling the time series data generation for the prototype
SAMPLES_PER_DAY = 24
# 2 years of 24 hours on 365 days timesteps from 0 to 731
TIME = np.arange(2 * 
                 SAMPLES_PER_DAY * 
                 SYNTHETHIC_TIMESERIES_CONFIG['period_length'] + 1,
                  dtype=np.float32) / SAMPLES_PER_DAY
TS_META = (TIME,
           SYNTHETHIC_TIMESERIES_CONFIG['baseline'], 
           SYNTHETHIC_TIMESERIES_CONFIG["amplitude"],
           SYNTHETHIC_TIMESERIES_CONFIG["slope"],
           SYNTHETHIC_TIMESERIES_CONFIG["noise_level"]
           )

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

