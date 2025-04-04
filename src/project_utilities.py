import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import config
import datetime
import os
import re
import json

def export_distance_matrix(np_matrix, filename=config.SYN_EXPORT_DIST_MATRIX_NAME, 
                           method=config.DEFAULT_DISSIMILARITY):
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir,"..", "experiments", "distance_matrices")
    filename = f"{filename}_{method}_{date}"
    filepath = os.path.join(output_dir, filename)
    np.save(filepath, np_matrix)
    print(f"Distance matrix saved to: {filepath}")
        
        
# TODO: import_distance_matrix function
def import_distance_matrix(filename=config.SYN_EXPORT_DIST_MATRIX_NAME, 
                           method=config.DEFAULT_DISSIMILARITY,
                           date=None):
    '''Imports a distance matrix from the respective experiments folder and depending
    on if a specific date has not been passed it will retrieve the newest file or a
    specific one.'''
    
    if date is None:
    
        filename_without_date = f"{filename}_{method}_"
        dir_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "..", 
            "experiments", 
            "distance_matrices"
        )
        
        all_files = os.listdir(dir_path)
        matching_files = []
        date_pattern = re.compile(rf"^{re.escape(filename_without_date)}(\d{{4}}-\d{{2}}-\d{{2}})\.npy$")
        
        for file in all_files:
            match = date_pattern.match(file)
            if match:
                file_date = match.group(1)
                matching_files.append((file_date, file))
        
        if not matching_files:
            raise FileNotFoundError(f"No distance matrix files found starting \
                                    with '{filename_without_date}' and ending with '{dir_path}' .")
                
        matching_files.sort(key=lambda x: x[0], reverse=True)
        _, youngest_filename = matching_files[0]
        filepath = os.path.join(dir_path, youngest_filename)
    else:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "..", 
                                "experiments", 
                                "distance_matrices", 
                                f"{filename}_{method}_{date}.npy")
    
    print(f"Loaded distance matrix from: {filepath}")

    return np.load(filepath)
    


def compute_and_save_accuracy(df, method_name):
    '''Computes the accuracy of how similar the given dataset to the uncorrupted 
    synthetic dataset is. After computation if prints out the result and exports 
    it into the log files in the experiments folder.'''
    ts_demo_data_clean = import_dataframe_from_csv(config.SYN_EXPORT_DATA_NAME+"_clean")
    #df = deindex_dataframe(df)
    values = []
    mse_value = mean_squared_error(ts_demo_data_clean['Value'], df['value'])
    print(f"The Mean-Squared-Error(MSE) for using the {method_name}-method is: \n{mse_value}")
    values.append(mse_value)
    
    mape_value = mean_absolute_percentage_error(ts_demo_data_clean['Value'], df['value'])
    print(f"The Mean-Absolute-Percentage-Error(MAPE) for using the {method_name}-method is: \n{mape_value}")
    values.append(mape_value)

    export_logfile(values, method_name)

    
def export_logfile(values, method_name):

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir,"..", "experiments", "logs", "interpolations")
    
    filename = os.path.join(output_dir, f"accuracy_log_{method_name}_{date}.json")
    
    with open(filename, "w", encoding='utf-8') as f: 
        json.dump({"method": method_name, 
                   "mse": values[0], 
                   "mape": values[-1], 
                   "date": date
                   }, 
                  f, 
                  ensure_ascii=False, 
                  indent=4)

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
    
def import_dataframe_from_csv_indexed(filename=config.SYN_EXPORT_DATA_NAME, restored=False):
    '''imports time series dataframe from csv and indexes the time column, 
    in order to leverage powerful and efficient DateTimeIndex functionality from pandas library'''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if restored:
        output_dir = os.path.join(script_dir,"..", "data", "restored")
    else:
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

def plot_time_series_comparison(series_dict, title="TimeSeries_Plot",
                                output_dir=None, 
                                xlabel="Time", ylabel="Value", fmt='-', 
                                freq='H'):
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
    
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "experiments", "plots")
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{title}_{date}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved to: {filepath}")
