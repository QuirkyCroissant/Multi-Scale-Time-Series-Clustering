import numpy as np

def trend(time, slope=0):
    return slope * time

def pattern_characteristics(season_time):
    '''adds seasonality pattern to the time series data in a cosine pattern'''
    pattern = np.where(season_time < 0.4,
                       np.cos(season_time * 2 * np.pi),
                       1 / np.exp(3 * season_time))
    return pattern

def seasonality(time, period, amplitude=1, phase=0):
    '''adds seasonality/repetitive pattern to the time series data'''
    season_time = ((time + phase) % period) / period
    return amplitude * pattern_characteristics(season_time)
    

def noise(time, noise_level=1, seed=None):
    '''adds noise to the time series data'''
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
    


def create_time_series(ts_metastats: tuple):
    '''main entrypoint in generating synthetic time series data'''
    time, baseline, amplitude, slope, noise_level = ts_metastats

    straight_series = baseline + trend(time, slope)
    seasonal_series = straight_series + seasonality(time, period=365, amplitude=amplitude)
    ts_series = seasonal_series + noise(time, noise_level, 69)

    return ts_series
    