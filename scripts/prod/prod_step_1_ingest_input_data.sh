#!/bin/bash
cd "$(dirname "$0")/../.."

start_time=$(date +%s)

TIMESTAMP_LOG=$(date +"%Y-%m-%d_%H-%M-%S")
LOGFILE="scripts/prod/logs/prod_step_1_ingest_input_data_$TIMESTAMP_LOG.txt"

N_TIMESERIES=100

{
    echo "=== STEP 1: Preprocess and Restore $N_TIMESERIES Time Series from Production Data ==="

    python3 src/main.py prod \
        --restore \
        --gen_amount $N_TIMESERIES

    echo "=== Step 1 completed: $N_TIMESERIES production time series preprocessed and restored. ==="

    end_time=$(date +%s)
    runtime=$((end_time - start_time))

    hours=$((runtime / 3600))
    minutes=$(( (runtime % 3600) / 60 ))
    seconds=$((runtime % 60))

    if (( hours > 0 )); then
        echo "Script finished in $hours hour(s), $minutes minute(s), and $seconds second(s)."
    elif (( minutes > 0 )); then
        echo "Script finished in $minutes minute(s) and $seconds second(s)."
    else
        echo "Script finished in $seconds second(s)."
    fi
} 2>&1 | tee "$LOGFILE"