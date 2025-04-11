import pandas as pd
import numpy as np
import config

def corrupt_timeunit(tunit_df, probability: dict):
    '''Driven by chance a time unit(eg.: an hour or day) will either be partialy deleted, 
    completely reduced to one smaller time unit measurement or will stay intact'''
    action = np.random.choice(
        ["intact", "partial", "reduce"],
        p=[probability["intact"], probability["partial"], probability["reduce"]]
    )

    if action == "intact":
        return tunit_df
    elif action == "partial":
        # randomly selects between 10 and 50% of hour measurements that will get deleted
        n = len(tunit_df)
        if n < 2:
            return tunit_df
        num_to_purge = np.random.randint(int(0.1*n), int(0.5*n) + 1)
        purge_indices = np.random.choice(tunit_df.index, size = num_to_purge, replace=False)
        return tunit_df.drop(purge_indices)
    
    elif action == "reduce":
        # returns only one measurement of the time unit (e.g. midnight measure for a day)
        return tunit_df.iloc[[0]]




def corrupt_time_series_data(ts_df: pd.DataFrame):
    '''Groups DateTimeIndexed dataframe into days and applies corruption function on each day'''
    probabilities = config.CORRUPTION_PROBS
    ts_df = ts_df.copy()

    inferred_frequency = pd.infer_freq(ts_df.index[:10])
    if inferred_frequency is None:
        raise ValueError("Could not infer frequency of time series data.")
    
    if inferred_frequency.startswith("D"):
        n_total = len(ts_df)
        n_corrupted = int(n_total * config.DAILY_CORRUPTION_RATE)
        corrupted_indices = np.random.choice((ts_df.copy()).index, size=n_corrupted, replace=False)
        return ts_df.drop(corrupted_indices)
    
    else:
        grouped = ts_df.groupby(ts_df.index.date)   
        corrupted_df = grouped.apply(lambda group: corrupt_timeunit(group, probabilities))
        return corrupted_df
    