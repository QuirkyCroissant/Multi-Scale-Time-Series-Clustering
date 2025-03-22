import pandas as pd
import numpy as np
import config

def corrupt_timeunit(tunit_df, probability: dict):
    '''Driven by chance a day will either be partialy deleted, 
    completely reduced to one hour measurement or will stay intact'''
    action = np.random.choice(
        ["intact", "partial", "reduce"],
        p=[probability["intact"], probability["partial"], probability["reduce"]]
    )

    if action == "intact":
        return tunit_df
    elif action == "partial":
        # randomly selects between 10 and 50% of hour measurements that will get deleted
        n = len(tunit_df)
        num_to_purge = np.random.randint(int(0.1*n), int(0.5*n) + 1)
        purge_indices = np.random.choice(tunit_df.index, size = num_to_purge, replace=False)
        return tunit_df.drop(purge_indices)
    
    elif action == "reduce":
        # returns only one measurement of the time unit (e.g. midnight measure for a day)
        return tunit_df.iloc[[0]]




def corrupt_time_series_data(ts_df: pd.DataFrame):
    '''Groups DateTimeIndexed dataframe into days and applies corruption function on each day'''
    probabilities = config.CORRUPTION_PROBS
    corrupted_df =ts_df.groupby(ts_df.index.date).apply(lambda x: corrupt_timeunit(x, probabilities))
    return corrupted_df
    