#!/bin/bash
cd "$(dirname "$0")/.."

echo "WARNING: This will delete all generated, corrupted, restored, and experiment result files."
read -p "Are you sure you want to proceed? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "Deleting generated and processed data files..."

find data/generated/ -type f -name 'ts_demo_data_*_clean' -exec echo "Deleting: {}" \; -exec rm -f {} +
find data/corrupted/ -type f -name 'ts_demo_data_*_corrupted' -exec echo "Deleting: {}" \; -exec rm -f {} +
find data/restored/ -mindepth 2 -type f ! -path "*/legacy/*" \
    ! -name '.gitignore' ! -name 'README.md' \
    -exec echo "Deleting: {}" \; -exec rm -f {} +

echo "Data files removed."

read -p "Do you also want to delete log files? (y/N): " delete_logs
if [[ "$delete_logs" =~ ^[Yy]$ ]]; then
    echo "Deleting log files..."
    find experiments/logs/clustering/default/ -type f ! -path "*/legacy/*" -exec echo "Deleting: {}" \; -exec rm -f {} +
    find experiments/logs/clustering/graph/ -type f ! -path "*/legacy/*" -exec echo "Deleting: {}" \; -exec rm -f {} +
    find experiments/logs/interpolations/ -type f ! -path "*/legacy/*" -exec echo "Deleting: {}" \; -exec rm -f {} +
    echo "Log files removed."
else
    echo "Skipped log file deletion."
fi

read -p "Do you also want to delete extracted production series files? (y/N): " delete_prod_series
if [[ "$delete_prod_series" =~ ^[Yy]$ ]]; then
    echo "Deleting extracted production time series..."
    find data/production_input/extracted_series/ -type f -name 'ts_prod_data_*_raw' -exec echo "Deleting: {}" \; -exec rm -f {} +
    echo "Extracted production series removed."
else
    echo "Skipped deletion of extracted production series."
fi

read -p "Do you also want to delete distance matrix files? (y/N): " delete_dist
if [[ "$delete_dist" =~ ^[Yy]$ ]]; then
    echo "Deleting distance matrices..."
    find experiments/distance_matrices/ -type f ! -path "*/legacy/*" -exec echo "Deleting: {}" \; -exec rm -f {} +
    echo "Distance matrices removed."
else
    echo "Skipped distance matrices deletion."
fi

read -p "Do you also want to delete plot files? (y/N): " delete_plots
if [[ "$delete_plots" =~ ^[Yy]$ ]]; then
    echo "Deleting plot files..."
    find experiments/plots/clustering/ -type f ! -path "*/legacy/*" -exec echo "Deleting: {}" \; -exec rm -f {} +
    find experiments/plots/comparisons/ -type f ! -path "*/legacy/*" -exec echo "Deleting: {}" \; -exec rm -f {} +
    find experiments/plots/corrupted_data/ -type f ! -path "*/legacy/*" -exec echo "Deleting: {}" \; -exec rm -f {} +
    find experiments/plots/generated_data/ -type f ! -path "*/legacy/*" -exec echo "Deleting: {}" \; -exec rm -f {} +
    find experiments/plots/graph_analysis/ -type f ! -path "*/legacy/*" -exec echo "Deleting: {}" \; -exec rm -f {} +
    find experiments/plots/interpolations/ -type f ! -path "*/legacy/*" -exec echo "Deleting: {}" \; -exec rm -f {} +
    echo "Plot files removed."
else
    echo "Skipped plot file deletion."
fi

read -p "Do you also want to delete evaluation files? (y/N): " delete_eval
if [[ "$delete_eval" =~ ^[Yy]$ ]]; then
    echo "Deleting evaluation files..."
    find experiments/evaluations/interpolations/ -type f ! -path "*/legacy/*" -exec echo "Deleting: {}" \; -exec rm -f {} +
    echo "Evaluation files removed."
else
    echo "Skipped evaluation file deletion."
fi

echo "Cleanup complete."