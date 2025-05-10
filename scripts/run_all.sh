#!/bin/bash
cd "$(dirname "$0")"

echo ""
echo " _______ _____ __  __ ______    _____ ______ _____  _____ ______  _____" 
echo "|__   __|_   _|  \/  |  ____|  / ____|  ____|  __ \|_   _|  ____|/ ____|"
echo "   | |    | | | \  / | |__    | (___ | |__  | |__) | | | | |__  | (___  "
echo "   | |    | | | |\/| |  __|    \___ \|  __| |  _  /  | | |  __|  \___ \ "
echo "   | |   _| |_| |  | | |____   ____) | |____| | \ \ _| |_| |____ ____) |"
echo "   |_|  |_____|_|  |_|______| |_____/|______|_|  \_\_____|______|_____/"
echo ""
echo "          TIME SERIES CLUSTERING - FULL PIPELINE EXECUTION"
echo ""

start_time=$(date +%s)

run_step () {
    STEP_NAME=$1
    SCRIPT=$2

    echo "--------------------------------------------------------------------------------"
    echo ">>> STARTING: $STEP_NAME"
    step_start=$(date +%s)

    bash "$SCRIPT"

    step_end=$(date +%s)
    duration=$((step_end - step_start))
    
    echo "COMPLETED: $STEP_NAME (Duration: ${duration}s)"
    
}

run_step "Step 1 - Generate, Corrupt, and Restore Time Series" "step_1_generate_corrupt_restore.sh"
run_step "Step 2 - Compute Distance Matrices" "step_2_compute_distance_matrices.sh"
run_step "Step 3 - Cluster and Analyze Results" "step_3_cluster_and_analyse.sh"

end_time=$(date +%s)
total_runtime=$((end_time - start_time))
minutes=$((total_runtime / 60))
seconds=$((total_runtime % 60))

echo "=== FULL EXPERIMENTS COMPLETED SUCCESSFULLY ==="
echo "Total runtime: ${minutes} minute(s) and ${seconds} second(s)"
echo ""
