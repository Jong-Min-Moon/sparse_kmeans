# ------------------------------------------------------------------
# sim_gaussian_unknowncov.R
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
p_seq      <- c(500, 1000, 2000, 3000, 4000, 5000)
noise_type <- "Gaussian"
methods <- c("scvx") # methods to benchmark

# Setup directories & logging
log_dir <- "logs"
res_dir <- "results/gaussian"

if (!dir.exists(log_dir)) dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(res_dir)) dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(log_dir, "sim_gaussian_unknowncov.log")
if (!file.exists(log_file)) file.create(log_file)

cat(sprintf("Starting Gaussian Unknown-Covariance Simulation. Results will be saved in %s/\n", res_dir))
log_progress(log_file, sprintf(
    "\n==========================================\n--- Gaussian Unknown-Covariance Started at %s ---\n==========================================\n",
    Sys.time()
))

# Setup parallel backend
n_cores <- parallel::detectCores() - 1
if (n_cores < 1) n_cores <- 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Pre-load utilities and packages on workers to avoid redundant IO
invisible(clusterEvalQ(cl, {
    source("sim_utils.R")
}))

log_progress(log_file, sprintf("Registered doParallel backend with %d cores.\n", n_cores))

overall_start_time <- Sys.time()

for (p in p_seq) {
    cat(sprintf("\n--- Dimension p = %d ---\n", p))
    log_progress(log_file, sprintf("\n--- Dimension p = %d ---\n", p))

    # Run simulations and collect log messages to avoid file contention
    log_msgs <- foreach(
        job_id    = 1:n_runs,
        .packages = c(
            "clue", "sparcl", "MASS", "methods",
            "scvxclustr", "cvxclustr", "igraph", "Matrix", "cluster",
            "mclust", "clustvarsel"
        )
    ) %dopar% {
        # 1. Skip if already completed
        if (check_progress(res_dir, job_id, p, methods = methods)) {
            return(NULL)
        }

        # 2. Reproducibility
        current_seed <- 2025 + job_id * 1000 + p

        # 3. Generate data under unknown covariance
        data_res <- tryCatch(
            generate_data_unknowncov(
                n = n, p = p, sep = separation,
                seed = current_seed, noise_type = noise_type
            ),
            error = function(e) {
                return(sprintf("Data Generation Error in rep %d, p = %d: %s\n", job_id, p, e$message))
            }
        )
        if (is.character(data_res)) return(data_res)
        if (is.null(data_res)) return(NULL)

        # 4. Evaluate competitor methods (scvx only as approved)
        sim_out <- run_simulation_methods(
            X           = data_res$X,
            true_labels = data_res$true_labels,
            K           = 2,
            p           = p,
            n           = n,
            sep         = separation,
            rho         = 0.45, 
            job_id      = job_id,
            seed        = current_seed,
            methods     = methods
        )

        # 5. Checkpoint
        save_result(sim_out$res_df, res_dir, job_id, p)

        # 6. Return log message for sequential writing
        return(sim_out$log_msg)
    }

    # Write collected logs sequentially to avoid contention
    for (msg in log_msgs) {
        if (!is.null(msg)) log_progress(log_file, msg)
    }
}

stopCluster(cl)

overall_runtime <- as.numeric(difftime(Sys.time(), overall_start_time, units = "mins"))
log_progress(log_file, sprintf(
    "\n==========================================\n--- Gaussian Unknown-Cov Simulation Completed at %s (Runtime: %.2f mins) ---\n==========================================\n",
    Sys.time(), overall_runtime
))
cat(sprintf("Execution completed in %.2f minutes.\n", overall_runtime))
