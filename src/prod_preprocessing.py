import pandas as pd
import numpy as np
import os

from project_utilities import (import_dataframe_from_csv,
                               export_dataframe_to_csv)
import config

def preprocess_prod_data_to_series():
    '''Reads production dataset(s) and splits each row (a time series) into
    a separate CSV file compliant with the application's downstream processing.'''

    prod_input_dir = config.TO_PROD_INPUT_DATA_DIR
    prod_output_dir = config.TO_PROD_SERIES_EXPORT_DATA_DIR

    total_series_count = 0
    for file in sorted(os.listdir(prod_input_dir)):
        if not file.endswith(".csv"):
            continue

        full_path = os.path.join(prod_input_dir, file)
        print(f"Processing file: {file}")

        curr_prod_df = pd.read_csv(full_path, sep=config.PROD_DATA_DELIMITER)
        if curr_prod_df.empty or curr_prod_df.shape[1] < 2:
            print(f"Skipping {file}: insufficient data.")
            continue

        # 1st column is series ID, rest are dates
        date_columns = pd.to_datetime(curr_prod_df.columns[1:], format=config.PROD_INPUT_TIMESTAMP_FORMAT)

        for row_idx, (_, row) in enumerate(curr_prod_df.iterrows()):
            series_df = pd.DataFrame({
                "time": date_columns,
                "value": row.values[1:]
            })
            output_filename = f"{config.PROD_EXPORT_DATA_NAME}_{total_series_count}_raw"
            export_dataframe_to_csv(
                series_df,
                output_filename,
                prod_output_dir,
                prod=True
            )
            total_series_count += 1

        print(f"Extracted {row_idx+1} series from {file}")

    print(f"Completed preprocessing production data: {total_series_count} total series extracted.")