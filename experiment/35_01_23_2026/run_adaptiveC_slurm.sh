#!/bin/bash

# --- 1. Configuration ---
BASE_DIR="/home1/jongminm/sparse_kmeans/experiment/35_01_23_2026"
RESULTS_DIR="$BASE_DIR/results/adaptiveC"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# --- 2. Parameters ---
C_MIN=0.1
C_MAX=0.6
P_VAL=8000
N_ITER=1000

# --- 3. Create SLURM array script ---
cat > "$BASE_DIR/sub_array.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=bandit_adaptiveC
#SBATCH --output=$LOG_DIR/job_%a.out
#SBATCH --error=$LOG_DIR/job_%a.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=1-100%20

module load legacy/CentOS7
module load matlab/2022a

cd "$BASE_DIR"
matlab -batch "run_bandit_adaptiveC_task(\${SLURM_ARRAY_TASK_ID}, $C_MIN, $C_MAX, $P_VAL, $N_ITER, '$RESULTS_DIR')"
EOF

# --- 4. Optional Merge Script ---
cat > "$BASE_DIR/sub_merge.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=merge_bandit_adaptiveC
#SBATCH --output=$LOG_DIR/merge.out
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --mem=4G

module load legacy/CentOS7
module load matlab/2022a

cd "$BASE_DIR"
matlab -batch "merge_adaptiveC_results('$RESULTS_DIR')"
EOF

# --- 5. Submit Jobs ---
echo "Submitting Simulation Array..."
JOB_ID=$(sbatch --parsable "$BASE_DIR/sub_array.sh")
echo "Submitted Array Job ID: $JOB_ID"
echo "Submitting Merge Job (after array completes)..."
sbatch --dependency=afterok:$JOB_ID "$BASE_DIR/sub_merge.sh"

echo "------------------------------------------------"
echo "Check progress with: squeue -u $USER"
echo "Logs are located in: $LOG_DIR"
echo "------------------------------------------------"
