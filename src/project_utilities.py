import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pandas as pd
import config
import datetime
import os

def compute_and_save_accuracy(df, method_name):
    '''Computes the accuracy of how similar the given dataset to the uncorrupted 
    synthetic dataset is. After computation if prints out the result and exports 
    it into the log files in the experiments folder.'''
    ts_demo_data_clean = import_dataframe_from_csv(config.SYN_EXPORT_DATA_NAME+"_clean")
    #df = deindex_dataframe(df)
    messages = []
    mse_value = mean_squared_error(ts_demo_data_clean['Value'], df['value'])
    mse_msg = str(f"The Mean-Squared-Error(MSE) for using the {method_name}-method is: \n{mse_value}")
    messages.append(mse_msg)
    print(mse_msg)
    
    mape_value = mean_absolute_percentage_error(ts_demo_data_clean['Value'], df['value'])
    mape_msg = str(f"The Mean-Absolute-Percentage-Error(MAPE) for using the {method_name}-method is: \n{mape_value}")
    messages.append(mape_msg)
    print(mape_msg)

    export_logfile(messages, method_name)

    
def export_logfile(messages, method_name):

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir,"..", "experiments", "logs")
    
    filename = os.path.join(output_dir, f"accuracy_log_{method_name}_{date}.txt")
    
    with open(filename, "w") as f:
        for message in messages:
            f.write(message + "\n")
    
    print(f"Log file saved to: {filename}")
    
    

def export_dataframe_to_csv(df, filename=config.SYN_EXPORT_DATA_NAME, output_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(script_dir,"..", "data")
    filepath = os.path.join(output_dir, filename)

    df.to_csv(filepath, index=False, sep=';')
    print(f"DataFrame exported to: {filepath}")
    
def import_dataframe_from_csv(filename=config.SYN_EXPORT_DATA_NAME, input_dir=None):
    '''imports time series dataframe from csv without indexing'''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if input_dir is None:
        input_dir = os.path.join(script_dir,"..", "data")
    filepath = os.path.join(input_dir, filename)

    df = pd.read_csv(filepath, sep=';')
    return df
    
def import_dataframe_from_csv_indexed(filename=config.SYN_EXPORT_DATA_NAME):
    '''imports time series dataframe from csv and indexes the time column, 
    in order to leverage powerful and efficient DateTimeIndex functionality from pandas library'''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir,"..", "data")
    filepath = os.path.join(output_dir, filename)

    df = pd.read_csv(filepath, sep=';', index_col=[0], parse_dates=[0])
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
