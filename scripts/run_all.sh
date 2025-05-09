#!/bin/bash
cd "$(dirname "$0")"

echo "=== RUNNING FULL TIME SERIES EXPERIMENTS ==="

bash step_1_generate_corrupt_restore.sh
bash step_2_compute_distance_matrices.sh
bash step_3_cluster_and_analyse.sh

echo "=== Full Experiments completed successfully. ==="
