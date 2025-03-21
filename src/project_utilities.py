import matplotlib.pyplot as plt
import datetime
import os

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