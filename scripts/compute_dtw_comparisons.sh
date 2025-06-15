#!/bin/bash
cd "$(dirname "$0")/.."

start_time=$(date +%s)

echo "=== SEQUENTIAL DISTANCE MATRIX + CLUSTERING PIPELINE (FIXED k=4) ==="

RESTORE_METHOD="spline"
N_TIMESERIES=100
CLUSTER_METHOD="kmedoids"
CLUSTER_K=4

# DTW (baseline)
echo "--- [DTW] Computing distance matrix ---"
python3 src/main.py prod \
    --distance \
    --dist_method dtw \
    --restore_method $RESTORE_METHOD \
    --gen_amount $N_TIMESERIES

echo "--- [DTW] Running clustering (k=$CLUSTER_K) ---"
python3 src/main.py prod \
    --cluster_method $CLUSTER_METHOD \
    --dist_method dtw \
    --restore_method $RESTORE_METHOD \
    --gen_amount $N_TIMESERIES \
    --cluster_k $CLUSTER_K

# fastDTW with varying radii
RADII=(1 2 3 4 5 6 7 8 9 10 20 30)

for R in "${RADII[@]}"; do
    echo "--- [fastDTW - radius $R] Computing distance matrix ---"
    python3 src/main.py prod \
        --distance \
        --dist_method fastDTW \
        --dist_radius $R \
        --restore_method $RESTORE_METHOD \
        --gen_amount $N_TIMESERIES

    echo "--- [fastDTW - radius $R] Running clustering (k=$CLUSTER_K) ---"
    python3 src/main.py prod \
        --cluster_method $CLUSTER_METHOD \
        --dist_method fastDTW \
        --dist_radius $R \
        --restore_method $RESTORE_METHOD \
        --gen_amount $N_TIMESERIES \
        --cluster_k $CLUSTER_K

    python3 -c "import gc; gc.collect()"
    echo "--- Done with fastDTW radius $R ---"
    echo ""
    sleep 5
done

echo "=== ALL COMPUTATIONS COMPLETE ==="

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
