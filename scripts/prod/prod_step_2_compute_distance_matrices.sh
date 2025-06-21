#!/bin/bash
cd "$(dirname "$0")/../.."

TIMESTAMP_LOG=$(date +"%Y-%m-%d_%H-%M-%S")
LOGFILE="scripts/prod/logs/prod_step_2_compute_distance_matrices_$TIMESTAMP_LOG.txt"

{
    start_time=$(date +%s)

    echo "=== STEP 2: Compute Distance Matrices ==="

    N_TIMESERIES=100
    METHODS=("dtw" "fastDTW" "pearson")
    AGGREGATION_METHOD="spline"


    for METHOD in "${METHODS[@]}"; do
        echo "--- Computing distances using method: $METHOD ---"
        
        # Pearson: only without normalization
        if [[ "$METHOD" == "pearson" ]]; then    
            python3 src/main.py prod \
                --distance \
                --dist_method "$METHOD" \
                --restore_method "$AGGREGATION_METHOD" \
                --gen_amount $N_TIMESERIES

        # fastDTW: using radius 3 is standard        
        elif [[ "$METHOD" == "fastDTW" ]]; then
            python3 src/main.py prod \
                --distance \
                --dist_method "$METHOD" \
                --dist_radius 3 \
                --restore_method "$AGGREGATION_METHOD" \
                --gen_amount $N_TIMESERIES

            python3 src/main.py prod \
                --distance \
                --dist_method "$METHOD" \
                --dist_radius 3 \
                --restore_method "$AGGREGATION_METHOD" \
                --normalized \
                --gen_amount $N_TIMESERIES

        else
            # both raw and normalized runs
            python3 src/main.py prod \
                --distance \
                --dist_method "$METHOD" \
                --restore_method "$AGGREGATION_METHOD" \
                --gen_amount $N_TIMESERIES

            python3 src/main.py prod \
                --distance \
                --dist_method "$METHOD" \
                --restore_method "$AGGREGATION_METHOD" \
                --normalized \
                --gen_amount $N_TIMESERIES
        fi

        python3 -c "import gc; gc.collect()"
        echo "--- Done with method: $METHOD ---"
        sleep 5
    done

    echo "=== Step 2 completed: Distance matrices computed and saved. ==="

    end_time=$(date +%s)
    runtime=$((end_time - start_time))

    hours=$((runtime / 3600))
    minutes=$(((runtime % 3600) / 60))
    seconds=$((runtime % 60))

    if (( hours > 0 )); then
        echo "Script finished in $hours hour(s), $minutes minute(s), and $seconds second(s)."
    elif (( minutes > 0 )); then
        echo "Script finished in $minutes minute(s) and $seconds second(s)."
    else
        echo "Script finished in $seconds second(s)."
    fi
} 2>&1 | tee "$LOGFILE"