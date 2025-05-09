#!/bin/bash
cd "$(dirname "$0")/.."

echo "=== STEP 2: Compute Distance Matrices ==="

python src/main.py demo \
    --distance \
    --dist_method fastDTW \
    --dist_radius 3 \
    --gen_amount 100

echo "=== Step 2 completed: Distance matrices computed and saved. ==="
