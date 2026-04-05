# ------------------------------------------------------------------
# test_rep.R
# Test a single repetition of the legacy simulation environment.
# ------------------------------------------------------------------
source("sim_utils.R")

# Fixed parameters
n <- 200
p <- 3000
separation <- 4
job_id <- 1
seed <- 2025 + job_id * 1000 + p
noise_type <- "Laplace"

cat(sprintf("Testing Rep %d, p = %d in R %s...\n", job_id, p, getRversion()))

# 1. Generate Data
cat("Generating data...\n")
spec <- get_specification_identity(support = 1:10, separation = separation, dimension = p)
res <- generate_data_from_specification(specification = spec, n = n, seed = seed, noise = noise_type)
X <- t(res$X)
true_labels <- res$labels

# 2. Run simulation methods
cat("Running methods (Witten, Arias, IF-PCA, B-Clust, SCVX)...\n")
sim_out <- run_simulation_methods(
    X = X,
    true_labels = true_labels,
    K = 2,
    p = p,
    n = n,
    sep = separation,
    rho = 0,
    job_id = job_id,
    seed = seed
)

print(sim_out$res_df)
cat("\nTest completed successfully.\n")
