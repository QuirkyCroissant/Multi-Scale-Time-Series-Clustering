#!/bin/bash
cd "$(dirname "$0")/.."

echo "=== STEP 3: Run Clustering and Plot Results ==="

python src/main.py demo \
    --cluster_method kmedoids \
    --cluster_k 4 \
    --dist_method fastDTW \
    --normalized \
    --gen_amount 100

echo "=== Step 3 completed: Clustering done and results logged. ==="
