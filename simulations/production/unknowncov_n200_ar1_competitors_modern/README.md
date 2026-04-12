# AR(1) Precision Matrix Simulation Framework (unknowncov_n200_ar1)

This directory serves as the **new centralized standard** for evaluating clustered algorithms on the Sparse Symmetric Data Generator framework.

Previous simulations (`sim_15`, `sim_16`) hardcoded evaluation pipelines into isolated algorithm buckets, risking discrepancies across the pseudo-random parameter initialization of generative clusters.

### Core Differences
1. **Single Data Source**: For each replication, precisely ONE data matrix ($X$) is generated.
2. **True Fairness**: *Witten*, *Arias*, and *IF-PCA* are fed that exact identically seeded output. 
3. **Structured Storage**: Outputs are no longer disjoint algorithms-specific structures, but clean tidydata tuples (`p`, `n`, `sep`, `rho`, `accuracy_*`, `runtime_*`).

### Data Specification
This variant uses an **Erdős–Rényi graph** structure for the precision matrix, integrated via `get_specification_erdos_renyi` in `data_generator.R`. This replaces the standard chain graph used in the original `unknowncov_n200_chain` simulation.

## Architecture

- **`methods_wrapper.R`**: Standardizes outputs from `Witten`, `Arias`, and `IF-PCA`, safely executing them against $X$, transposing dimension requirements contextually, and mapping back the resulting group assignments alongside runtimes dynamically.
- **`accuracy_utils.R`**: Isolates performance mapping using the Hungarian Matching algorithm against base truth.
- **`run_simulation.R`**: The core local driver. Executes nested parallel nodes traversing independent replications across CPU threads safely.
- **`aggregate_results.R`**: Collapses multi-dimensional tracking objects into explicit parameter-driven analytical matrices.

## How to Run
Run across locally allocated cores:
```bash
Rscript run_simulation.R
```

Compile results:
```bash
Rscript aggregate_results.R
```
*Wait for aggregation to output `summary_sim17.csv` for human-readable tracking logs.*
