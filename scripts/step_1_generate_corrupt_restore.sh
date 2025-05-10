#!/bin/bash
cd "$(dirname "$0")/.."

start_time=$(date +%s)

echo "=== STEP 1: Generate, Corrupt, and Restore Time Series Data ==="

echo "=== Generate Time Series Data"

N_TIMESERIES=10

python3 src/main.py demo \
    --new_data \
    --comp_img \
    --restore \
    --gen_amount $N_TIMESERIES

echo "=== Step 1 completed: All data generated, corrupted, and restored. ==="

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
