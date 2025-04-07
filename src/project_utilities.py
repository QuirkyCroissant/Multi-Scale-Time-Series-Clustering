import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import zscore
import pandas as pd
import numpy as np
import config
import datetime
import os
import re
import json

def export_distance_matrix(np_matrix, 
                           filename=config.SYN_EXPORT_DIST_MATRIX_NAME, 
                           method=config.DEFAULT_DISSIMILARITY,
                           normalized=False):
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir,"..", "experiments", "distance_matrices")
    
    if normalized:
        filename = f"{filename}_normalized_{method}_{date}"
    else:
        filename = f"{filename}_raw_{method}_{date}"
    
    filepath = os.path.join(output_dir, filename)
    np.save(filepath, np_matrix)
    print(f"Distance matrix saved to: {filepath}")
        
        
# TODO: import_distance_matrix function
def import_distance_matrix(filename=config.SYN_EXPORT_DIST_MATRIX_NAME, 
                           is_normalize=False,
                           method=config.DEFAULT_DISSIMILARITY,
                           date=None):
    '''Imports a distance matrix from the respective experiments folder and depending
    on if a specific date has not been passed it will retrieve the newest file or a
    specific one.'''
    
    if date is None:
    
        if is_normalize:
            filename_without_date = f"{filename}_normalized_{method}_"
        else:
            filename_without_date = f"{filename}_raw_{method}_"

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
            error_msg = f"No distance matrix files found starting with '{filename_without_date}' and ending with '{dir_path}' ."
            raise FileNotFoundError(error_msg)
                
        matching_files.sort(key=lambda x: x[0], reverse=True)
        _, youngest_filename = matching_files[0]
        filepath = os.path.join(dir_path, youngest_filename)

    else:
        if is_normalize:
            filename_with_date = f"{filename}_normalized_{method}_{date}.npy"
        else:
            filename_with_date = f"{filename}_raw_{method}_{date}.npy"

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "..", 
                                "experiments", 
                                "distance_matrices", 
                                filename_with_date)
    
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
    
    

def export_dataframe_to_csv(df, 
                            filename=config.SYN_EXPORT_DATA_NAME, 
                            output_dir=None,
                            clean=False,
                            corrupted=False):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        if clean:
            output_dir = os.path.join(script_dir,"..", "data", "generated")
        elif corrupted:
            output_dir = os.path.join(script_dir,"..", "data", "corrupted")
        else:
            output_dir = os.path.join(script_dir,"..", "data")
            
    if clean:
        filename = f"{filename}_clean"
    elif corrupted:
        filename = f"{filename}_corrupted"
        

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
        output_dir = os.path.join(script_dir,"..", "data", "generated")
    filepath = os.path.join(output_dir, filename)

    df = pd.read_csv(filepath, sep=';', index_col=[0], parse_dates=[0])
    return df
    
def deindex_dataframe(dataframe):
    return dataframe.reset_index(level=0, drop=True).reset_index().rename(columns={'Time': 'time', 'Value': 'value'})

def plot_time_series(y, x=None, format='-', start=0, end=None,
                     title=None, xlabel=None, ylabel=None, 
                     legend=None, output_dir=None):
    plt.figure(figsize=(10, 6))

    if x is None:
        x = range(len(y))

    if isinstance(y, (list, np.ndarray)) and y.ndim == 1:
        plt.plot(x[start:end], y[start:end], format)
    elif isinstance(y, tuple):
        for y_i in y:
            plt.plot(x[start:end], y_i[start:end], format)
    else:
        raise ValueError("Unsupported format for y")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if legend:
        plt.legend(legend)

    plt.title(title)
    plt.grid(True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(script_dir, "..", "experiments", "plots", "generated_data")

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

def plot_silhouette_score(k_values, silhoutte_scores, is_normalized=False):
    '''creates silhoutte score plot and saves it into the experiments folder'''
    plt.figure()
    plt.plot(k_values, silhoutte_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters (Normalized)' 
              if is_normalized else 
              'Silhouette Score vs. Number of Clusters')
    plt.grid(True)

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    if is_normalized:
        filename = f"silhouette_score_plot_normalized_{date}.png"
    else:
        filename = f"silhouette_score_plot_raw_{date}.png"
        
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..",
                             "experiments",
                             "plots",
                             "clustering",
                             filename
                             )
    plt.savefig(plot_path)
    print(f"Silhouette score plot saved to: {plot_path}")
    plt.close()

def plot_kmedoid_results(time_series, 
                         segmented_sequences, 
                         labels, 
                         model, 
                         is_normalized=False,
                         segment_length=config.SEGMENTATION_WINDOW):

    n_clusters = len(np.unique(labels))
    colors = plt.cm.get_cmap("tab10", n_clusters)
    hours = np.arange(segment_length)

    _, axs = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

    ### Full Time Series coloured by cluster ###
    for i, label in enumerate(labels):
        start = i * segment_length
        end = start + segment_length
        if end > len(time_series):
            break
        segment = time_series[start:end]
        axs[0].plot(range(start, end), segment, color=colors(label), linewidth=0.8)
    
    if is_normalized:
        plot_header = "Full Time Series with Segments Colored by Cluster (Normalized)"
    else:
        plot_header = "Full Time Series with Segments Colored by Cluster"

    axs[0].set_title(plot_header)
    axs[0].set_xlabel("Time (Hours or Days)")
    axs[0].set_ylabel("Value")
    axs[0].grid(True)


    ### Medoid sequences ###
    for i, idx in enumerate(model.medoid_indices_):
        axs[1].plot(hours, segmented_sequences[idx], label=f"Medoid {i}", color=colors(i), linewidth=2.5)

    axs[1].set_title("Cluster Medoid Profiles (Normalized)" 
                     if is_normalized else 
                     "Cluster Medoid Profiles"
                     )
    axs[1].set_xlabel("Hour of Day")
    axs[1].set_ylabel("Value")
    axs[1].legend()
    axs[1].grid(True)



    date = datetime.datetime.now().strftime("%Y-%m-%d")
    if is_normalized:
        filename = f"kmedoid_results_plot_normalized_{date}.png"
    else:
        filename = f"kmedoid_results_plot_raw_{date}.png"
        

    plot_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "experiments", "plots", "clustering",
        filename
    )

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Combined clustering plot saved to: {plot_path}")
