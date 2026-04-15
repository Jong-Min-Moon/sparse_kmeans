# ------------------------------------------------------------------
# sim_gaussian_ar1_conda.R
# scvxclustr Evaluation under Gaussian Noise (AR1 Setting)
# ------------------------------------------------------------------
library(foreach)
library(doParallel)

# Ensure script runs in its own directory
args <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0 && file.exists(script_name)) {
    setwd(dirname(normalizePath(script_name)))
}

source("sim_utils.R")

# Fixed parameters mimicking unknowncov_n200_ar1_competitors_modern
n <- 200
n_runs <- 100
separation <- 6
p_seq      <- c(1000, 2000, 3000, 4000, 5000)
noise_type <- "Gaussian"
methods <- c("scvx")

# Setup directories & logging
log_dir <- "logs"
res_dir <- "results_ar1/gaussian"

if (!dir.exists(log_dir)) dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(res_dir)) dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(log_dir, "sim_gaussian_ar1_conda.log")
if (!file.exists(log_file)) file.create(log_file)

cat(sprintf("Starting Gaussian AR1 Simulation for scvxclustr. Results will be saved in %s/\n", res_dir))
log_progress(log_file, sprintf(
    "\n==========================================\n--- Gaussian AR1 Started at %s ---\n==========================================\n",
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

    log_msgs <- foreach(
        job_id    = 1:n_runs,
        .packages = c(
            "clue", "sparcl", "MASS", "methods",
            "scvxclustr", "cvxclustr", "igraph", "Matrix", "cluster",
            "mclust", "clustvarsel"
        )
    ) %dopar% {
        if (check_progress(res_dir, job_id, p, methods = methods)) {
            return(NULL)
        }

        current_seed <- 2025 + job_id * 1000 + p

        data_res <- tryCatch(
            generate_data_ar1_conda(
                n = n, p = p, sep = separation,
                seed = current_seed, noise_type = noise_type
            ),
            error = function(e) {
                return(sprintf("Data Generation Error in rep %d, p = %d: %s\n", job_id, p, e$message))
            }
        )
        if (is.character(data_res)) return(data_res)
        if (is.null(data_res)) return(NULL)

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

        save_result(sim_out$res_df, res_dir, job_id, p)
        return(sim_out$log_msg)
    }

    for (msg in log_msgs) {
        if (!is.null(msg)) log_progress(log_file, msg)
    }
}

stopCluster(cl)

overall_runtime <- as.numeric(difftime(Sys.time(), overall_start_time, units = "mins"))
log_progress(log_file, sprintf(
    "\n==========================================\n--- Gaussian AR1 Simulation Completed at %s (Runtime: %.2f mins) ---\n==========================================\n",
    Sys.time(), overall_runtime
))
cat(sprintf("Execution completed in %.2f minutes.\n", overall_runtime))
