#!/bin/bash
cd "$(dirname "$0")/.."

echo "=== STEP 1: Generate, Corrupt, and Restore Time Series Data ==="

python src/main.py demo \
    --new_data \
    --comp_img \
    --restore \
    --gen_amount 100

echo "=== Step 1 completed: All data generated, corrupted, and restored. ==="
