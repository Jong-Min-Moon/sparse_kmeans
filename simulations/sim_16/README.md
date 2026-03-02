# Simulation 16: IF-PCA Clustering Evaluation

This directory contains the simulation setup to evaluate the **Influential Feature PCA (IF-PCA)** algorithm in R mapping directly to the data generating procedure used in Simulation 15 (`sim_15_witten_arias`).

The simulation setting matches the Symmetric Data Generator exactly:
- `p = 400`
- `n = 200`
- `K = 2`
- `rho = 0.45`
- Cluster separation: `4` or `5`

## Scripts

- `driver.R`: The core R script that generates data and runs IF-PCA clustering for a single replication (identified by `--job_id` and `--sep`).
- `run_local.R`: An R script designed to run the 100 iterations per separation level locally in parallel using `doParallel`.
- `aggregate_sim16.R`: An R script to aggregate all iteration results and compute mean/sd metrics for clustering accuracy and the average number of features selected (`L`).

## Workflow

### 1. Generating Results Locally
Because these IF-PCA runs are highly efficient locally and optimized via `eigen`/Gram matrices, HPC distribution is not required. You can launch all 100 iterations for both separations in parallel locally:
```bash
Rscript run_local.R
```
This automatically partitions the problem across your machine's logical cores. Results for each simulation iteration are placed in `results/`.

### 2. Aggregation
Once the `results/` directory is fully populated with `sim_id{id}_sep{sep}.rds` files, collapse them dynamically into readable summaries with:
```bash
Rscript aggregate_sim16.R
```
This script computes averages across the 100 iterations per condition and yields:
- `aggregated_sim16.rds`: Clean tidy tabular data for every single run.
- `summary_sim16.rds`: Compressed data format summary.
- `summary_sim16.csv`: Easily readable grouped CSV with explicit Mean and Standard Deviation metrics for tracking performance across separations.
