import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from project_utilities import (import_dataframe_from_csv,
                               export_dataframe_to_csv)
import config


def preprocess_prod_data_to_series(limit=40):
    """Efficiently processes Smart Meter CSV files from London dataset into 1D time series per household.
    This version supports multiple input CSVs and applies a household limit for scalability."""

    prod_input_dir = config.TO_PROD_INPUT_DATA_DIR
    prod_output_dir = config.TO_PROD_SERIES_EXPORT_DATA_DIR

    total_exported = 0
    processed_ids = set()

    for file in sorted(os.listdir(prod_input_dir)):
        if not file.endswith(".csv"):
            continue

        full_path = os.path.join(prod_input_dir, file)
        print(f"Reading: {full_path}")
        chunk_iter = pd.read_csv(full_path, chunksize=100_000, na_values=["Null"])

        
        # Efficient chunked reading for large files
        for chunk in tqdm(chunk_iter, desc=f"Processing {file}"):
            
            chunk.columns = [col.strip() for col in chunk.columns]
            chunk = chunk.rename(columns={
                "LCLid": "household",
                "DateTime": "time",
                chunk.columns[3]: "value"
            })

            chunk["value"] = chunk["value"].astype(np.float32)

            chunk = chunk.dropna(subset=["value"])
            chunk["time"] = pd.to_datetime(chunk["time"])

            grouped = chunk.groupby("household")

            for household_id, group in grouped:
                if household_id in processed_ids:
                    continue
                if limit and total_exported >= limit:
                    print(f"Limit of {limit} series reached.")
                    return

                group = group.sort_values("time")
                series_df = pd.DataFrame({
                    "time": group["time"].values,
                    "value": group["value"].values.astype(np.float32)
                })

                output_filename = f"{config.PROD_EXPORT_DATA_NAME}_{total_exported}_raw"
                export_dataframe_to_csv(series_df, output_filename, prod_output_dir, prod=True)

                print(f"[{total_exported}] Saved household {household_id} to {output_filename}")
                processed_ids.add(household_id)
                total_exported += 1

    print(f"Completed preprocessing production data: {total_exported} total household time series exported.")

"""def preprocess_prod_data_to_series():
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
"""