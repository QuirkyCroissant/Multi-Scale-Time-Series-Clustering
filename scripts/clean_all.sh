#!/bin/bash
cd "$(dirname "$0")/.."

echo "WARNING: This will delete all generated, corrupted, and restored data."
read -p "Are you sure you want to proceed? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "Deleting generated and processed data files..."

for file in data/generated/ts_demo_data_*_clean.csv; do
    [ -e "$file" ] && echo "Deleting: $file" && rm "$file"
done

for file in data/corrupted/ts_demo_data_*_corrupted.csv; do
    [ -e "$file" ] && echo "Deleting: $file" && rm "$file"
done

find data/restored/ -mindepth 1 -maxdepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +

echo "Data files removed."

read -p "Do you also want to delete log files? (y/N): " delete_logs
if [[ "$delete_logs" =~ ^[Yy]$ ]]; then
    echo "Deleting log files..."
    find experiments/logs/clustering/default/ -mindepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +
    find experiments/logs/clustering/graph/ -mindepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +
    find experiments/logs/interpolations/ -mindepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +
    echo "Log files removed."
else
    echo "Skipped log file deletion."
fi

read -p "Do you also want to delete distance matrix files? (y/N): " delete_dist
if [[ "$delete_dist" =~ ^[Yy]$ ]]; then
    echo "Deleting distance matrices..."
    find experiments/distance_matrices/ -mindepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +
    echo "Distance matrices removed."
else
    echo "Skipped distance matrices deletion."
fi

read -p "Do you also want to delete plot files? (y/N): " delete_plots
if [[ "$delete_plots" =~ ^[Yy]$ ]]; then
    echo "Deleting plot files..."
    find experiments/plots/clustering/ -mindepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +
    find experiments/plots/comparisons/ -mindepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +
    find experiments/plots/corrupted_data/ -mindepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +
    find experiments/plots/generated_data/ -mindepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +
    find experiments/plots/graph_analysis/ -mindepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +
    find experiments/plots/interpolations/ -mindepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +
    echo "Plot files removed."
else
    echo "Skipped plot file deletion."
fi

read -p "Do you also want to delete evaluation files? (y/N): " delete_eval
if [[ "$delete_eval" =~ ^[Yy]$ ]]; then
    echo "Deleting evaluation files..."
    find experiments/evaluations/interpolations/ -mindepth 1 ! -name legacy -exec echo "Deleting: {}" \; -exec rm -rf {} +
    echo "Evaluation files removed."
else
    echo "Skipped evaluation file deletion."
fi

echo "Cleanup complete."
