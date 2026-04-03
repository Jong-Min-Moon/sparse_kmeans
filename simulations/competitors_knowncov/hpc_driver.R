# ------------------------------------------------------------------
# hpc_driver.R
# Single-Replicate Driver for Competitor Evaluation (HPC)
# ------------------------------------------------------------------
# This script executes one replicate of the simulation study.
# It is designed to be called via Slurm array jobs.
# ------------------------------------------------------------------

library(methods)
library(MASS)
library(clue)
library(sparcl)

# 1. Parse Command Line Arguments
args <- commandArgs(trailingOnly = TRUE)

job_id   <- 1
p        <- 100
sep      <- 4
noise    <- "Gaussian"
methods_to_run <- c("witten", "arias", "ifpca", "cvs") # SCVX excluded

if (length(args) > 0) {
    for (i in seq_along(args)) {
        if (args[i] == "--job_id" && i < length(args)) job_id <- as.integer(args[i+1])
        if (args[i] == "--p"      && i < length(args)) p      <- as.integer(args[i+1])
        if (args[i] == "--sep"    && i < length(args)) sep    <- as.numeric(args[i+1])
        if (args[i] == "--noise"  && i < length(args)) noise  <- args[i+1]
        if (args[i] == "--methods" && i < length(args)) {
            methods_to_run <- strsplit(args[i+1], ",")[[1]]
        }
    }
}

cat(sprintf("--- HPC Driver: Job %d, p=%d, sep=%.1f, noise=%s ---\n", job_id, p, sep, noise))

# 2. Source Dependencies
# Note: On HPC, paths must be relative to the simulation directory
if (file.exists("../competitors_unknowncov/sim_utils.R")) {
    old_dir <- setwd("../competitors_unknowncov")
    source("sim_utils.R")
    setwd(old_dir)
} else {
    # Fallback for local testing or different root
    source("../../code_r/data_generator.R")
    source("../../code_r/methods_wrapper.R")
}

# 3. Define Known-Covariance Data Generation (Identity)
generate_data_knowncov <- function(n, p, sep, seed, noise_type) {
    support <- 1:10
    spec <- get_specification_identity(
        support = support,
        separation = sep,
        dimension = p
    )
    res <- generate_data_from_specification(
        specification = spec,
        n = n,
        seed = seed,
        noise = noise_type
    )
    # Competitor wrappers expect features in columns (n x p)
    # but the generator returns (n x p).
    # Wait, sim_utils.R: generate_data returns list(X = t(res$X), ...)
    # Let's check sim_utils.R again. 
    # sim_utils.R: generate_data <- function(...) { ... return(list(X = t(res$X), ...)) }
    # This means X is (p x n). 
    # HOWEVER, methods_wrapper.R: run_simulation_methods(X, ...)
    # says @param X Data matrix (n x p).
    # Let's check sim_gaussian_knowncov.R again.
    # It does: return(list(X = t(res$X), ...)) and then passes data_res$X to run_simulation_methods.
    # This implies run_simulation_methods expects (p x n)? 
    # Let me re-read methods_wrapper.R.
    # Line 53 in methods_wrapper.R: @param X Data matrix (n x p)
    # Line 98 in methods_wrapper.R: IF-PCA expects features in rows (p x n); transpose from standard (n x p)
    # Line 100: X_ifpca <- t(X)
    # So run_simulation_methods expects (n x p).
    # If generate_data_from_specification returns (n x p), then t(res$X) is (p x n).
    # If the local script sim_gaussian_knowncov.R uses t(res$X), then it's passing (p x n).
    # This is a potential discrepancy in the existing code I should be careful about.
    # Actually, in most of my previous fixes, I standardized to (n x p).
    
    return(list(X = t(res$X), true_labels = res$labels, spec = spec))
}

# 4. Execute Simulation
set.seed(2026 + job_id * 100 + p)
n <- 200 # Standard sample size

data_res <- tryCatch({
    generate_data_knowncov(n = n, p = p, sep = sep, seed = 2026 + job_id, noise_type = noise)
}, error = function(e) {
    stop("Data generation failed: ", e$message)
})

# Run competitor methods
sim_out <- run_simulation_methods(
    X = data_res$X,
    true_labels = data_res$true_labels,
    K = 2,
    p = p,
    n = n,
    sep = sep,
    rho = 0,
    job_id = job_id,
    seed = 2026 + job_id,
    methods = methods_to_run
)

# 5. Save Results
out_dir <- file.path("results_raw", tolower(noise), sprintf("p%d", p))
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_file <- file.path(out_dir, sprintf("sim_job%d_p%d.rds", job_id, p))
saveRDS(sim_out$res_df, file = out_file)

cat(sim_out$log_msg)
cat(sprintf("Result saved to %s\n", out_file))
