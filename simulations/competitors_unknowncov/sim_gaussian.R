# ------------------------------------------------------------------
# sim_gaussian.R
# Competitor Evaluation under Gaussian Noise
# ------------------------------------------------------------------
library(foreach)
library(doParallel)

# Ensure we are in the script's directory (if run via Rscript)
args <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0 && file.exists(script_name)) {
    setwd(dirname(normalizePath(script_name)))
}

source("sim_utils.R")

# Fixed parameters as per instructions
n <- 200
n_runs <- 100
separation <- 6
p_seq <- seq(50, 500, by = 50)
noise_type <- "Gaussian"

# Setup Directories & Logging
log_dir <- "logs"
res_dir <- "results/gaussian"
ensure_dir(log_dir)
ensure_dir(res_dir)

log_file <- file.path(log_dir, "sim_gaussian.log")

cat(sprintf("Starting Gaussian Simulation. Results will be saved in %s/\n", res_dir))
log_progress(log_file, sprintf("\n==========================================\n--- Gaussian Simulation Started at %s ---\n==========================================\n", Sys.time()))

# Setup parallel backend
n_cores <- parallel::detectCores() - 1
if (n_cores < 1) n_cores <- 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)
log_progress(log_file, sprintf("Registered doParallel backend with %d cores.\n", n_cores))

# Export local environment to workers
overall_start_time <- Sys.time()

for (p in p_seq) {
    cat(sprintf("\n--- Dimension p = %d ---\n", p))
    log_progress(log_file, sprintf("\n--- Dimension p = %d ---\n", p))
    
    # foreach for parallel execution over runs
    results <- foreach(
        job_id = 1:n_runs,
        .packages = c("clue", "sparcl", "MASS", "methods")
    ) %dopar% {
        
        # Reload utility scripts inside the worker to ensure paths and env
        source("sim_utils.R")
        
        # 1. Skip if already completed (Checkpointing)
        if (check_progress(res_dir, job_id, p)) {
            return(NULL) # Run exists
        }
        
        # 2. Strict Reproducibility (combining job_id and p ensures uniqueness)
        current_seed <- 2025 + job_id * 1000 + p
        
        # 3. Generate Data
        data_res <- tryCatch({
            generate_data(n = n, p = p, sep = separation, seed = current_seed, noise_type = noise_type)
        }, error = function(e) {
            log_progress(log_file, sprintf("Data Generation Error in rep %d, p = %d: %s\n", job_id, p, e$message))
            return(NULL)
        })
        
        if (is.null(data_res)) return(NULL)
        
        # 4. Evaluate Competitor Methods
        sim_out <- run_simulation_methods(
            X = data_res$X,
            true_labels = data_res$true_labels,
            K = 2,
            p = p,
            n = n,
            sep = separation,
            rho = 0.45,
            job_id = job_id,
            seed = current_seed
        )
        
        # 5. Checkpoint
        save_result(sim_out$res_df, res_dir, job_id, p)
        
        # 6. Logging
        log_progress(log_file, sim_out$log_msg)
        
        return(NULL)
    }
}

stopCluster(cl)

overall_runtime <- as.numeric(difftime(Sys.time(), overall_start_time, units = "mins"))
log_progress(log_file, sprintf("\n==========================================\n--- Gaussian Simulation Completed at %s (Runtime: %.2f mins) ---\n==========================================\n", Sys.time(), overall_runtime))
cat(sprintf("Execution completed in %.2f minutes.\n", overall_runtime))
