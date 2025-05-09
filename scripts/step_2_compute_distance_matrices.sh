#!/bin/bash
cd "$(dirname "$0")/.."

start_time=$(date +%s)

echo "=== STEP 2: Compute Distance Matrices ==="

python3 src/main.py demo \
    --distance \
    --dist_method fastDTW \
    --dist_radius 3 \
    --gen_amount 100

echo "=== Step 2 completed: Distance matrices computed and saved. ==="

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