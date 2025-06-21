#!/bin/bash
cd "$(dirname "$0")/../.."

TIMESTAMP_LOG=$(date +"%Y-%m-%d_%H-%M-%S")
LOGFILE="scripts/prod/logs/prod_step_3_clustering_$TIMESTAMP_LOG.txt"

start_time=$(date +%s)



METHODS=("dtw" "fastDTW")
CLUSTERING_METHODS=("kmedoids" "hierarchical")
NORMALIZED=("false" "true")
K_OPTIONS=(0 4)

RESTORE_METHOD="spline"
N_TIMESERIES=100

{
    echo "=== STEP 3: Run Clustering Experiments ==="

    for METHOD in "${METHODS[@]}"; do
        for NORM in "${NORMALIZED[@]}"; do

            for CLUSTER in "${CLUSTERING_METHODS[@]}"; do
                for K in "${K_OPTIONS[@]}"; do

                    echo "--- Clustering: $CLUSTER | Dissimilarity: $METHOD | Normalized: $NORM | k: $K ---"

                    CMD="python3 src/main.py prod \
                        --restore_method $RESTORE_METHOD \
                        --dist_method $METHOD \
                        --cluster_method $CLUSTER"

                    # Radius only needed for fastDTW
                    if [[ "$METHOD" == "fastDTW" ]]; then
                        CMD="$CMD --dist_radius 3"
                    fi

                    if [[ "$NORM" == "true" ]]; then
                        CMD="$CMD --normalized"
                    fi

                    if [[ "$K" -eq 0 ]]; then
                        CMD="$CMD --optimize_k"
                    else
                        CMD="$CMD --cluster_k $K"
                    fi

                    eval $CMD

                    echo "--- Done: $CLUSTER | $METHOD | Normalized: $NORM | k: $K ---"
                    echo ""
                done
            done
        done
    done

    echo "=== STEP 3.5: Graph-Based Clustering with Pearson Distance ==="

    GRAPH_METHODS=("louvain" "modularity" "label") 

    for METHOD in "${GRAPH_METHODS[@]}"; do
        echo "--- Running graph clustering: $METHOD ---"

        python3 src/main.py prod \
            --restore_method $RESTORE_METHOD \
            --dist_method pearson \
            --cluster_method $METHOD

        echo "--- Done: $METHOD ---"
        echo ""
    done

    echo "=== Step 3 completed: Clustering results generated. ==="

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