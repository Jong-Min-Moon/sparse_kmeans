# Simulation 14: Thompson Sampling Bandit Optimization (Parameter Sweeps)

Evaluates the `block_coordinate_optim_thompson` logic operating strictly on raw `p_val_thresholds` natively (rejecting pseudo-FDR wrappers).

### Settings mapped
- $n = 200, p = 400, K = 2$
- `rho = 0.45`, `support = 1:10`
- **Separation Levels**: [4, 5]
- **Raw P-Value Target**: [0.01, 0.005, 0.001]
- **Replications**: 20 Jobs / Grid Cell

### HPC Deployment
Data executes on Discovery seamlessly deploying local assets iterating Slurm blocks over `$SEP` and `$PVAL`.
1. Deploy explicitly via Powershell Terminal:
```powershell
./deploy_sim14.ps1
```
2. Pull outputs remotely to `$LocalDir\results_raw`:
```powershell
./retrieve_sim14.ps1
```
3. Process aggregated metrics targeting `results_aggregated\summary_sim14.csv`:
```powershell
Rscript aggregate_results.R
```
