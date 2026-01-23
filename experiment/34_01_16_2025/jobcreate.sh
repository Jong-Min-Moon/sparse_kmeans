#!/bin/bash

# --- 1. Configuration ---
BASE_DIR="/home1/jongminm/sparse_kmeans/experiment/34_01_16_2025"
DB_PATH="/home1/jongminm/sparse_kmeans/sparse_kmeans.db"
TABLE_NAME="bandit_C_streamlined"

PARAM_FILE="$BASE_DIR/param_list.txt"
RESULTS_DIR="$BASE_DIR/results_mat"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# --- 2. Generate Parameter Map ---
C_VALS=(0.1 0.01)
P_VALS=(8000)
REPS=$(seq 1 100)

echo "Generating parameter map..."
> "$PARAM_FILE"
for c in "${C_VALS[@]}"; do
    for p in "${P_VALS[@]}"; do
        for r in $REPS; do
            echo "$c $p $r" >> "$PARAM_FILE"
        done
    done
done

TOTAL_JOBS=$(wc -l < "$PARAM_FILE")
echo "Total combinations: $TOTAL_JOBS"

# --- 3. Create the SLURM Simulation Script ---
cat > "$BASE_DIR/sub_array.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=bandit_sim
#SBATCH --output=$LOG_DIR/job_%a.out
#SBATCH --error=$LOG_DIR/job_%a.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=23:59:59
#SBATCH --array=1-${TOTAL_JOBS}%50

module load legacy/CentOS7
module load matlab/2022a

cd "$BASE_DIR"
matlab -batch "run_bandit_task(\${SLURM_ARRAY_TASK_ID}, '$PARAM_FILE', '$RESULTS_DIR')"
EOF

# --- 4. Create the Cleanup/Merge Script ---
cat > "$BASE_DIR/sub_merge.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=merge_bandit
#SBATCH --output=$LOG_DIR/merge.out
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --mem=4G

module load legacy/CentOS7

module load matlab/2022a
cd "$BASE_DIR"
matlab -batch "merge_results_to_sqlite('$RESULTS_DIR', '$DB_PATH', '$TABLE_NAME')"
EOF

# --- 5. Submit the Jobs ---
echo "Submitting Simulation Array..."
# Use --parsable to get just the job number (e.g., 123456)
JOB_ID=$(sbatch --parsable "$BASE_DIR/sub_array.sh")

if [ -z "$JOB_ID" ]; then
    echo "Error: Job submission failed."
    exit 1
fi

echo "Submitted Array Job ID: $JOB_ID"
echo "Submitting Merge Job (waiting for Array Job $JOB_ID to finish)..."

# This job will stay in 'Dependency' status until the array job finishes successfully
sbatch --dependency=afterok:$JOB_ID "$BASE_DIR/sub_merge.sh"

echo "------------------------------------------------"
echo "Check progress with: squeue -u $USER"
echo "Logs are located in: $LOG_DIR"
echo "------------------------------------------------"