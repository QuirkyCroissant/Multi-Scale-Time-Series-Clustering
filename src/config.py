import numpy as np

SYNTHETHIC_TIMESERIES_CONFIG: dict = {
  "period_length": 365,
  "baseline": 10,
  "amplitude": 40,
  "slope": 0.05,
  "noise_level": 5
}

# Constants that are controlling the time series data generation for the prototype
TIME = np.arange(2 * SYNTHETHIC_TIMESERIES_CONFIG['period_length'] + 1, dtype=np.float32)
TS_META = (TIME,
           SYNTHETHIC_TIMESERIES_CONFIG['baseline'], 
           SYNTHETHIC_TIMESERIES_CONFIG["amplitude"],
           SYNTHETHIC_TIMESERIES_CONFIG["slope"],
           SYNTHETHIC_TIMESERIES_CONFIG["noise_level"]
           )

SYN_EXPORT_TITLE = "Synthetic Time Series"
RANDOM_SEED = 69