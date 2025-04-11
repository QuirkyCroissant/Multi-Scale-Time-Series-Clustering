import numpy as np
import pandas as pd
import config

def trend(time, slope=0):
    return slope * time

def daily_pattern(time, amplitude=1.0, shift=0):
    '''adds daily sinusoidal pattern to the time series data
    (simulating morning and evening peaks)'''
    return amplitude * np.sin(2 * np.pi * (time % 1 ) + shift)

def weekly_pattern(time, amplitude=1.0):
    '''adds weekly pattern to the synthetic data, boosting values
      on the end of a given week'''
    days = (time % 7).astype(int)
    weekend_boosted = (days >= 5).astype(float)
    return amplitude * weekend_boosted
    
def holiday_boost(time, amplitude=1.0, start_day=224, duration=62):
    '''All days in the series which are bigger than the start_day and 
    in the specified duration get boosted'''
    return amplitude * ((time >= start_day) & (time < start_day + duration)).astype(float)
    

def noise(time, noise_level=1, seed=None):
    '''adds noise to the time series data'''
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
    
def convert_time_series_to_dataframe(ts_series: np.ndarray, start_date=config.TIME_SERIES_START_DATE):
    '''Converts a 1D time series array into a dataframe with daily datetime index'''

    n_points = len(ts_series)
    date_index = pd.date_range(start=start_date, periods=n_points, freq="D")

    return  pd.DataFrame(
        {"time": date_index,"value": ts_series}
    )
    
def create_individual_time_series(time, 
                            baseline=10, 
                            daily_amp=1.0, 
                            weekly_amp=0.0, 
                            holiday_amp=0.0,
                            slope=0.1, 
                            noise_level=1.0, 
                            seed=config.RANDOM_SEED):
    '''Creates an individual time series which adhears to multiple patterns and trends.'''
    daily = daily_pattern(time, amplitude=daily_amp)
    weekly = weekly_pattern(time, amplitude=weekly_amp)
    holiday = holiday_boost(time, amplitude=holiday_amp)
    linear_trend = trend(time, slope=slope)
    noise_term = noise(time, noise_level=noise_level, seed=seed)

    return baseline + daily + weekly + holiday + linear_trend + noise_term


def create_multiple_time_series(n=config.AMOUNT_OF_INDIVIDUAL_SERIES, 
                                time=None, 
                                seed=config.RANDOM_SEED):
    '''Creates an array of multiple time series with the provided profiles from the config file.'''
    np.random.seed(seed)
    subjects = []
    if time is None:
        days = config.PERIOD_LENGTH
        time = np.arange(days, dtype=np.float32)
    
    profile_names = list(config.SYN_SERIES_PROFILES.keys())
    
    for i in range(n):
        profile_type = profile_names[i % len(profile_names)]
        params = config.SYN_SERIES_PROFILES[profile_type]
        subjects.append(create_individual_time_series(time, seed=seed + i, **params))
    
    return np.array(subjects)
        
    