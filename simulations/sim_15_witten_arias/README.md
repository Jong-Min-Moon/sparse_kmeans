# Simulation 15: Witten and Arias-Castro Sparse K-Means

This directory contains the simulation setup to evaluate two modernized (refactored) versions of legacy sparse K-Means algorithms:
1. **Witten's Sparse K-Means** (L1/L2 penalized)
2. **Arias-Castro's Sparse K-Means** (Hill climbing)

The simulation setting matches `sim14` (Symmetric Data Generator) with the following parameters:
- `p = 400`
- `n = 200`
- `K = 2`
- `rho = 0.45`
- Cluster separation: `4` or `5`

## Scripts

- `driver.R`: The core R script that generates data and runs both clustering methods for a single replication (identified by `--job_id`).
- `run_local.R`: An R script designed to run the 100 iterations per separation level locally in parallel using `doParallel`.
- `submit.sh` & `deploy_sim15.ps1`: HPC deployment and submission scripts (kept as alternative).
- `retrieve_sim15.ps1`: A PowerShell script for retrieving the `.rds` output files from the cluster.
- `aggregate_sim15.R`: An R script to aggregate all iteration results and compute mean/sd metrics for clustering accuracy.

## Workflow

### 1. Running Simulations locally (Recommended)
You can directly run all jobs locally, across `sep = 4` and `sep = 5` via multithreading using:
```bash
Rscript run_local.R
```

### 1b. Submitting Jobs (Alternative HPC method)
On the remote HPC cluster, navigate to this directory and submit the SLURM job array:
```bash
# For separation = 4
sbatch submit.sh 4

# For separation = 5
sbatch submit.sh 5
```

### 2. Retrieving Results
On your local machine, run the PowerShell retrieval script to download the `results/` folder:
```powershell
.\retrieve_sim15.ps1
```

### 3. Aggregation
Once the `results/` directory is populated locally, run the aggregation script:
```bash
Rscript aggregate_sim15.R
```
This will produce `aggregated_sim15.rds` (all raw results), `summary_sim15.rds`, and a readable `summary_sim15.csv` containing the mean and standard deviation of clustering accuracies for both separation levels.
