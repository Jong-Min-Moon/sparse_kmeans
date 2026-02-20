# Simulation Driver for HPC
# Usage: Rscript simulation_driver.R --job_id <ID> --n_iter <N> --config <JSON_FILE>

# Parse arguments manually to avoid dependencies (optparse)
args <- commandArgs(trailingOnly = TRUE)

# Default values
job_id <- 1
n_iter <- 100
config_file <- NULL
out_dir <- "results"

# Parse args
if (length(args) > 0) {
  for (i in seq(1, length(args), by=2)) {
    arg <- args[i]
    val <- args[i+1]
    
    if (arg == "--job_id") {
      job_id <- as.integer(val)
    } else if (arg == "--n_iter") {
      n_iter <- as.integer(val)
    } else if (arg == "--config") {
      config_file <- val
    } else if (arg == "--out_dir") {
      out_dir <- val
    }
  }
}

# Only load jsonlite if config is provided
if (!is.null(config_file)) {
    if (!require(jsonlite)) {
        stop("Package 'jsonlite' is required for config file usage. Please install it or run without --config.")
    }
}

# Variables for script (mapping opt$name to variable)
# To minimize code changes below, we can set specific variables
opt <- list(
    job_id = job_id,
    n_iter = n_iter,
    config = config_file,
    out_dir = out_dir
)

# Source Core Functions
# Assuming code is in 'code_r' relative to script, or script is in 'code_r'
# Adjust paths as needed based on deployment structure
# If running FROM 'code_r':
source("sdp_kmeans_bandit_unknowncov.R")
source("block_coordinate_optim_thompson.R")
# Dependencies sourced inside these files:
# source("sdp_kmeans.R")
# source("utils.R")
# source("clustering_block_unknowncov.R")
# source("clustering_block_knowncov.R")
# source("selection_block_greedy_screening.R")
# source("cluster_spectral.R")
# source("ISEE_bicluster.R") 
# source("get_cov_small.R") 

# Set Seed based on Job ID
set.seed(opt$job_id)

# Configuration Defaults
# You can load these from a JSON file if --config is provided
sim_config <- list(
  n = 100,
  p = 50,
  K = 2,
  signal_strength = 3,
  covariance_type = "identity" # or "toeplitz", "block"
)

if (!is.null(opt$config)) {
  if (file.exists(opt$config)) {
    sim_config <- fromJSON(opt$config)
  } else {
    warning("Config file not found. Using defaults.")
  }
}

# Generate Data
# Simple synthetic generation for demo
n <- sim_config$n
p <- sim_config$p
K <- sim_config$K
mu_signal <- sim_config$signal_strength

# Signal on first 5 features
mu1 <- rep(0, p)
mu2 <- rep(0, p)
mu2[1:5] <- mu_signal

# Generate X based on Covariance Type
if (sim_config$covariance_type == "identity") {
  X1 <- matrix(rnorm(n/2 * p), nrow=n/2)
  X2 <- matrix(rnorm(n/2 * p) + matrix(rep(mu2, n/2), nrow=n/2, byrow=TRUE), nrow=n/2)
  X <- rbind(X1, X2)
  X <- t(X)
} else {
  # Implement other covariance structures here
  # For now, fallback to Identity
  warning("Only Identity covariance implemented in driver for now.")
  X1 <- matrix(rnorm(n/2 * p), nrow=n/2)
  X2 <- matrix(rnorm(n/2 * p) + matrix(rep(mu2, n/2), nrow=n/2, byrow=TRUE), nrow=n/2)
  X <- rbind(X1, X2)
  X <- t(X)
}

true_labels <- c(rep(1, n/2), rep(2, n/2))

# Run Simulation
# Using Unknown Covariance Bandit
# Adjust parameters as needed
cat(sprintf("Job %d: Starting Simulation...\n", opt$job_id))
start_time <- Sys.time()

res <- sdp_kmeans_bandit_unknowncov(X, K, n_iter = opt$n_iter, C = 0.5, FDR_level = 0.4, n_perms = 100)

end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units="secs"))

# Process Results
# Calculate Accuracy
acc <- max(mean(res$cluster == true_labels), mean(res$cluster != true_labels))
selected_correctly <- all(res$selected[1:5]) && !any(res$selected[6:p])

output <- list(
  job_id = opt$job_id,
  config = sim_config,
  result = res, # Save full result or just metrics? Full result can be large.
  metrics = list(
    accuracy = acc,
    runtime = runtime,
    selected_features = which(res$selected),
    selected_correctly = selected_correctly
  )
)

# Save Output
dir.create(opt$out_dir, showWarnings = FALSE)
saveRDS(output, file = file.path(opt$out_dir, sprintf("sim_res_%d.rds", opt$job_id)))

cat(sprintf("Job %d: Finished. Accuracy: %.4f\n", opt$job_id, acc))
