# HPC Workflow: Competitors Known-Covariance Simulation

This workflow automates the deployment, execution, and aggregation of sparse clustering competitor benchmarks on a High-Performance Computing (HPC) cluster using Slurm.

## Methodology
The simulation evaluates **Witten**, **Arias-Castro**, **IF-PCA**, and **clustvarsel** (SCVX is excluded) under various feature dimensions ($p$) and separations. Independent replications are parallelized as Slurm array jobs.

## Workflow Phases

### 1. Deploy Phase
Prepares the HPC environment, transfers dependencies, compiles C++ utilities, and submits jobs.

```powershell
# Default: Gaussian noise, p = 50, 100, ..., 500
.\deploy_knowncov.ps1 -Username your_hpc_user

# Custom: Laplace noise, high dimensions
.\deploy_knowncov.ps1 -Username your_hpc_user -Noise "Laplace" -Dimensions @(3000, 6000, 9000, 12000)
```

### 2. Retrieve Phase
Downloads raw results (`.rds`) and logs from the HPC to the local `results_raw/` directory.

```powershell
.\retrieve_knowncov.ps1 -Username your_hpc_user
```

### 3. Aggregate Phase
Combines the hundreds/thousands of individual result files into cohesive summary tables.

```bash
Rscript aggregate_hpc.R
```

## Directory Structure
- `hpc_driver.R`: Executes a single simulation replicate.
- `submit.sh`: Slurm job template.
- `results_raw/`: Raw output from compute nodes (e.g., `results_raw/gaussian/p500/sim_job1_p500.rds`).
- `logs/`: Slurm error and output logs.
- `aggregated_hpc_[noise].rds`: Monolithic dataset for all replicates.
- `summary_hpc_[noise].csv`: Tidy summary statistics (mean/SD).

## Requirements
- HPC with Slurm scheduler.
- R (recommended: 4.5.1) with packages: `dplyr`, `tidyr`, `purrr`, `MASS`, `clue`, `sparcl`, `mclust`.
- C++ compiler for `selection_utils.cpp` and `proj_simplex.cpp`.
