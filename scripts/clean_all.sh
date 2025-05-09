#!/bin/bash
cd "$(dirname "$0")/.."

echo "WARNING: This will delete all generated, corrupted, and restored data."
read -p "Are you sure you want to proceed? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "Deleting generated and processed data files..."
rm -f data/generated/ts_demo_data_*_clean.csv
rm -f data/corrupted/ts_demo_data_*_corrupted.csv
rm -rf data/restored/*/

echo "Data files removed."

read -p "Do you also want to delete log files? (y/N): " delete_logs
if [[ "$delete_logs" =~ ^[Yy]$ ]]; then
    echo "Deleting log files..."
    rm -rf experiments/logs/clustering/default/*
    rm -rf experiments/logs/clustering/graph/*
    rm -rf experiments/logs/interpolations/*
    echo "Log files removed."
else
    echo "Skipped log file deletion."
fi

read -p "Do you also want to delete plot files? (y/N): " delete_plots
if [[ "$delete_plots" =~ ^[Yy]$ ]]; then
    echo "Deleting plot files..."
    rm -rf experiments/plots/clustering/*
    rm -rf experiments/plots/comparisons/*
    rm -rf experiments/plots/corrupted_data/*
    rm -rf experiments/plots/generated_data/*
    rm -rf experiments/plots/graph_analysis/*
    rm -rf experiments/plots/interpolations/*
    echo "Plot files removed."
else
    echo "Skipped plot file deletion."
fi

echo "Cleanup complete."
