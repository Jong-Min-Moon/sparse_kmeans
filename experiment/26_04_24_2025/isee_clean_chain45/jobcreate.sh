#!/bin/bash

# Directories
BASE_DIR="/home1/jongminm/sparse_kmeans/experiment/26_04_24_2025/isee_clean_chain45"
MATLAB_TEMPLATE="$BASE_DIR/matlab_template.m"
DB_DIR="/home1/jongminm/sparse_kmeans/sparse_kmeans.db"
TABLE_NAME="isee_new"



for rep in $(seq 201 2000); do
    # Define filenames
    MFILE="$BASE_DIR/isee_rho45_sep4_rep_${rep}.m"
    JOBFILE="$BASE_DIR/isee_rho45_sep4_rep_${rep}.sh"
    OUTFILE="$BASE_DIR/isee_rho45_sep4_rep_${rep}.out"

    # === Create .m file ===
    cat > "$MFILE" <<EOF
pc = parallel.cluster.Local;
job_folder = fullfile('/home1/jongminm/.matlab/local_cluster_jobs/R2022a', getenv('SLURM_JOB_ID'));
if ~exist(job_folder, 'dir')
    mkdir(job_folder);
end
set(pc, 'JobStorageLocation', job_folder);
ncores = str2num(getenv('SLURM_CPUS_PER_TASK')) - 1;
pool = parpool(pc, ncores);

p = 400;
rep = ${rep};
sep = 4;
n = 500;
addpath(genpath('/home1/jongminm/sparse_kmeans'));

table_name = '${TABLE_NAME}';
db_dir = '${DB_DIR}';

model = 'chain45';
cluster_1_ratio = 0.5;

[data, label_true, mu1, mu2, sep, ~, beta_star] = generate_gaussian_data(n, p, 4, model, rep, cluster_1_ratio);
data = data';
label_true = label_true';

ISEE_kmeans_clean_simul(data, 2, 100, true, 10, 5, 0.01, db_dir, table_name, rep, model, sep, label_true);

delete(pool);
EOF

    # === Create SLURM job script ===
    cat > "$JOBFILE" <<EOF
#!/bin/bash
#SBATCH --output="${OUTFILE}"
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=23:59:59

echo "Starting job for rep=${rep} on \$(hostname) at \$(date)"

module purge
module load legacy/CentOS7
module load matlab/2022a

cd "$BASE_DIR"
matlab -batch isee_rho45_sep4_rep_${rep}

echo "Finished job for rep=${rep} at \$(date)"
EOF

    # === Submit job ===
    sbatch "$JOBFILE"
     sleep 1
done
