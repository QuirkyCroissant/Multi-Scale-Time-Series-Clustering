import matplotlib.pyplot as plt
import pandas as pd
import config
import datetime
import os

def export_dataframe_to_csv(df, filename=config.SYN_EXPORT_DATA_NAME):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir,"..", "data")
    filepath = os.path.join(output_dir, filename)

    df.to_csv(filepath, index=False, sep=';')
    print(f"DataFrame exported to: {filepath}")
    
def import_dataframe_from_csv(filename=config.SYN_EXPORT_DATA_NAME):
    '''imports time series dataframe from csv without indexing'''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir,"..", "data")
    filepath = os.path.join(output_dir, filename)

    df = pd.read_csv(filepath, sep=';')
    return df
    
def import_dataframe_from_csv_indexed(filename=config.SYN_EXPORT_DATA_NAME):
    '''imports time series dataframe from csv and indexes the time column, 
    in order to leverage powerful and efficient DateTimeIndex functionality from pandas library'''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir,"..", "data")
    filepath = os.path.join(output_dir, filename)

    df = pd.read_csv(filepath, sep=';', index_col=['Time'], parse_dates=['Time'])
    return df
    
def deindex_dataframe(dataframe):
    return dataframe.reset_index(level=0, drop=True).reset_index().rename(columns={'Time': 'time', 'Value': 'value'})

def plot_time_series(x, y, format='-', start=0, end=None,
                     title=None, xlabel=None, ylabel=None, legend=None ):
    plt.figure(figsize=(10, 6))

    # differentiates between one and multiple time series to plot
    if type(y) is tuple:
        for y_i in y:
            plt.plot(x[start:end], y_i[start:end], format)
    else:
        plt.plot(x[start:end], y[start:end], format)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend:
        plt.legend(legend)

    plt.title(title)
    plt.grid(True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir,"..", "experiments", "plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the file path
    filename = f"{title}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to: {filepath}")


def plot_time_series_comparison(series_dict, title="TimeSeries_Plot", xlabel="Time", ylabel="Value", fmt='-', freq='H'):
    '''Plots time series and exports the image, is possible to compare multiple series.'''

    plt.figure(figsize=(10, 6))

    for label, (x, y) in series_dict.items():
        # series get normalized by frequenzy(hourly, daily etc.)
        s = pd.Series(y.values, index=pd.to_datetime(x))
        new_index = pd.date_range(start=s.index.min(), end=s.index.max(), freq=freq)
        s_reindexed = s.reindex(new_index)
        # time series now gets plotted true to its temporal scale, NaN values get ignored
        plt.plot(s_reindexed.index, s_reindexed.values, fmt, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "experiments", "plots")
    
    filename = f"{title}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to: {filepath}")
