#!/bin/bash
cd "$(dirname "$0")/.."

start_time=$(date +%s)

echo "=== STEP 3: Run Clustering and Plot Results ==="

python3 src/main.py demo \
    --cluster_method kmedoids \
    --cluster_k 4 \
    --dist_method fastDTW \
    --normalized \
    --gen_amount 100

echo "=== Step 3 completed: Clustering done and results logged. ==="

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