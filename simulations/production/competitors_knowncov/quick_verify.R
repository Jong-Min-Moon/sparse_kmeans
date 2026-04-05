# ------------------------------------------------------------------
# quick_verify.R
# Small-scale simulation to verify SCVX grid search & pipeline
# ------------------------------------------------------------------
library(methods)

# 1. Setup paths
# Set working directory to script location
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) setwd(dirname(normalizePath(script_path)))

# Source utilities (which now point to code_r/methods_wrapper.R)
old_dir <- setwd("../competitors_unknowncov")
source("sim_utils.R")
setwd(old_dir)

# 2. Known-Covariance Data Generation (Identity)
generate_data_knowncov <- function(n, p, sep, seed, noise_type) {
    support <- 1:10
    spec <- get_specification_identity(support = support, separation = sep, dimension = p)
    res <- generate_data_from_specification(specification = spec, n = n, seed = seed, noise = noise_type)
    return(list(X = t(res$X), true_labels = res$labels, spec = spec))
}

# 3. Parameters
n_val <- 100
p_val <- 200
sep_val <- 10
seed_val <- 1234
noise_val <- "Laplace"

cat(sprintf("--- Quick Verification Run (p=%d, n=%d) ---\n", p_val, n_val))

# 4. Execute Simulation Step
set.seed(seed_val)
data_obj <- generate_data_knowncov(n_val, p_val, sep_val, seed_val, noise_val)

# Run all methods (including the new SCVX grid search)
cat("Running run_simulation_methods (this includes the SCVX grid search)...\n")
methods_out <- run_simulation_methods(
    X = data_obj$X,
    true_labels = data_obj$true_labels,
    K = 2,
    p = p_val,
    n = n_val,
    sep = sep_val,
    rho = 0,
    job_id = 1,
    seed = seed_val
)

# 6. Report
cat("\nResults Summary:\n")
cat(methods_out$log_msg)

if (!is.na(methods_out$res_df$accuracy_scvx)) {
    cat("\nSUCCESS: SCVX returned a non-NA accuracy. Pipeline integrated correctly.\n")
} else {
    cat("\nFAILURE: SCVX accuracy is NA. Check logs for Master Error.\n")
}
