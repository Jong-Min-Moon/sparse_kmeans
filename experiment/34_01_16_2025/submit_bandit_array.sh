#!/bin/bash

# ===============================
# 1. Configuration
# ===============================
BASE_DIR="/home1/jongminm/sparse_kmeans/experiment/34_01_16_2025"
RESULTS_DIR="$BASE_DIR/results_mat"
LOG_DIR="$BASE_DIR/logs"
PARAM_FILE="$BASE_DIR/param_list.txt"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ===============================
# 2. Generate Parameter Map
# ===============================
C_VALS=(0.2 0.4 0.6 0.8)
P_VALS=(8000)
REPS=$(seq 1 20)

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

# ===============================
# 3. Create the SLURM Array Script
# ===============================
cat > "$BASE_DIR/sub_array.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=bandit_sim
#SBATCH --output=$LOG_DIR/job_%a.out
#SBATCH --error=$LOG_DIR/job_%a.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=23:59:59
#SBATCH --array=1-${TOTAL_JOBS}%50

# Load modules
module load legacy/CentOS7
module load matlab/2022a

cd "$BASE_DIR"

# Run MATLAB task
matlab -batch "run_bandit_task(\${SLURM_ARRAY_TASK_ID}, '$PARAM_FILE', '$RESULTS_DIR')"
EOF

# ===============================
# 4. Submit the Array Job
# ===============================
echo "Submitting Simulation Array..."
JOB_ID=$(sbatch --parsable "$BASE_DIR/sub_array.sh")

if [ -z "$JOB_ID" ]; then
    echo "Error: Job submission failed."
    exit 1
fi

echo "Submitted Array Job ID: $JOB_ID"
echo "------------------------------------------------"
echo "Check progress with: squeue -u $USER"
echo "Logs are located in: $LOG_DIR"
echo "------------------------------------------------"
