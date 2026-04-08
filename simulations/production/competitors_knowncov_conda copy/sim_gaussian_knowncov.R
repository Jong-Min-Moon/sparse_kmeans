# ------------------------------------------------------------------
# sim_gaussian_knowncov.R
# Competitor Evaluation under Gaussian Noise (Known Covariance)
# ------------------------------------------------------------------
library(foreach)
library(doParallel)

# Ensure script runs in its own directory
args <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0 && file.exists(script_name)) {
    setwd(dirname(normalizePath(script_name)))
}

# Source local utilities (data generation + logging helpers + run_simulation_methods)
source("sim_utils.R")

# Fixed parameters
n <- 200
n_runs <- 100
separation <- 6
# p_seq      <- c(1000, seq(3000, 30000, by = 3000))
p_seq <- seq(50, 500, by = 50)
noise_type <- "Gaussian"
methods <- c("witten", "arias", "ifpca", "scvx", "cvs") # methods to benchmark

# Setup directories & logging
log_dir <- "logs"
res_dir <- "results/gaussian"

if (!dir.exists(log_dir)) dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(res_dir)) dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(log_dir, "sim_gaussian_knowncov.log")
if (!file.exists(log_file)) file.create(log_file)

cat(sprintf("Starting Gaussian Known-Covariance Simulation. Results will be saved in %s/\n", res_dir))
log_progress(log_file, sprintf(
    "\n==========================================\n--- Gaussian Known-Covariance Started at %s ---\n==========================================\n",
    Sys.time()
))

# Setup parallel backend
n_cores <- parallel::detectCores() - 1
if (n_cores < 1) n_cores <- 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)
log_progress(log_file, sprintf("Registered doParallel backend with %d cores.\n", n_cores))

overall_start_time <- Sys.time()

for (p in p_seq) {
    cat(sprintf("\n--- Dimension p = %d ---\n", p))
    log_progress(log_file, sprintf("\n--- Dimension p = %d ---\n", p))

    results <- foreach(
        job_id = 1:n_runs,
        .packages = c(
            "clue", "sparcl", "MASS", "methods",
            "scvxclustr", "cvxclustr", "igraph", "Matrix", "cluster",
            "mclust", "clustvarsel"
        )
    ) %dopar% {
        # Reload utilities inside worker (parallel environment isolation)
        source("sim_utils.R")

        # 1. Skip if already completed (validates method coverage)
        if (check_progress(res_dir, job_id, p, methods = methods)) {
            return(NULL)
        }

        # 2. Reproducibility
        current_seed <- 2025 + job_id * 1000 + p

        # 3. Generate data under identity covariance
        data_res <- tryCatch(
            generate_data_knowncov(
                n = n, p = p, sep = separation,
                seed = current_seed, noise_type = noise_type
            ),
            error = function(e) {
                log_progress(log_file, sprintf(
                    "Data Generation Error in rep %d, p = %d: %s\n", job_id, p, e$message
                ))
                NULL
            }
        )
        if (is.null(data_res)) {
            return(NULL)
        }

        # 4. Evaluate competitor methods
        sim_out <- run_simulation_methods(
            X           = data_res$X,
            true_labels = data_res$true_labels,
            K           = 2,
            p           = p,
            n           = n,
            sep         = separation,
            rho         = 0.45, # updated to 0.45 for unknown covariance
            job_id      = job_id,
            seed        = current_seed,
            methods     = methods
        )

        # 5. Checkpoint
        save_result(sim_out$res_df, res_dir, job_id, p)

        # 6. Log
        log_progress(log_file, sim_out$log_msg)

        return(NULL)
    }
}

stopCluster(cl)

overall_runtime <- as.numeric(difftime(Sys.time(), overall_start_time, units = "mins"))
log_progress(log_file, sprintf(
    "\n==========================================\n--- Gaussian Known-Cov Simulation Completed at %s (Runtime: %.2f mins) ---\n==========================================\n",
    Sys.time(), overall_runtime
))
cat(sprintf("Execution completed in %.2f minutes.\n", overall_runtime))
