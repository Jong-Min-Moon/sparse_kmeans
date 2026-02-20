# Parallel Benchmark Iterative SDP K-Means

# 1. Load Required Packages
local({r <- getOption("repos")
       r["CRAN"] <- "https://cloud.r-project.org" 
       options(repos=r)
})

# Make sure parallel packages are installed
required_packages <- c("CVXR", "MASS", "dplyr", "doParallel", "foreach")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg)
}

library(CVXR)
library(MASS)
library(dplyr)
library(doParallel)
library(foreach)

# 2. Simulation Parameters
p <- 3000                # Dimension
n_runs <- 100            # Replications
n_core <- 5              # Number of cores
center_dist <- 4         # L2 separation
N <- 200                 # Total observations
n1 <- 100; n2 <- 100     # Cluster sizes
k <- 2                   # Number of clusters
seed_offset <- 200000    # Ensure unique seeds from previous runs

# Setup Cluster
cat(sprintf("Setting up cluster with %d cores...\n", n_core))
cl <- makeCluster(n_core)
registerDoParallel(cl)

# 3. Simulation Functions
# We define everything inside the foreach or export needed functions.
# However, sourcing the files is easiest inside the worker loop.
# We'll need the path to the source files since workers might not start in the same WD by default depending on OS/config.
# On Windows, they often do not share WD.
# We will get the current script directory or assume getwd() is correct if launched from correct place.

current_wd <- getwd()
cat(sprintf("Current working directory: %s\n", current_wd))

# Export current WD to workers
clusterExport(cl, "current_wd")
clusterEvalQ(cl, {
  setwd(current_wd)
  if (!file.exists("sdp_kmeans_iter_knowncov.R")) {
     stop(paste("Cannot find source file in", getwd()))
  }
  source("sdp_kmeans_iter_knowncov.R")
})

# 4. Simulation Loop
cat(sprintf("Starting Parallel Simulation (p=%d, dist=%.1f, runs=%d)...\n", p, center_dist, n_runs))
total_start_time <- Sys.time()

results_log <- foreach(run = 1:n_runs, .combine = rbind, .packages = c("CVXR", "MASS", "dplyr")) %dopar% {
  
  # Ensure unique seed
  current_seed <- p + run + seed_offset
  set.seed(current_seed)
  
  # Define Sparse Cluster Centers
  mu1 <- rep(0, p)
  nonzero_count <- 10
  val_per_entry <- sqrt(center_dist^2 / nonzero_count)
  mu2 <- rep(0, p)
  mu2[1:nonzero_count] <- val_per_entry
  
  tryCatch({
    # --- A. Data Generation ---
    X1 <- matrix(rnorm(n1 * p) + mu1, nrow = n1, byrow = TRUE)
    X2 <- matrix(rnorm(n2 * p) + mu2, nrow = n2, byrow = TRUE)
    X <- t(rbind(X1, X2))
    true_labels <- c(rep(1, n1), rep(2, n2))
    
    # --- B. Run Iterative Algorithm ---
    start_run <- Sys.time()
    # Using stable_iter=5 and fdr_level=0.4
    result_iter <- sdp_kmeans_iter_knowncov(X, k, n_iter = 20, stable_iter = 5, fdr_level = 0.4)
    end_run <- Sys.time()
    runtime <- as.numeric(difftime(end_run, start_run, units="secs"))
    
    estimated_labels <- result_iter$cluster
    iterations <- result_iter$iter
    
    # --- C. Accuracy Calculation ---
    t1 = as.integer(true_labels)
    e1 = as.integer(estimated_labels)
    
    # Check if clusters collapsed
    if (length(unique(e1)) < 2) {
       # Failed clustering (all one cluster)
       match_acc <- max(table(t1)) / N 
    } else {
       acc_1 <- mean(t1 == e1)
       e2 <- ifelse(e1 == 1, 2, 1) # simple swap for k=2
       acc_2 <- mean(t1 == e2)
       match_acc <- max(acc_1, acc_2)
    }
    
    data.frame(
      Run_ID = run,
      Dimension = p,
      Center_Dist = center_dist,
      Seed = current_seed,
      Accuracy = match_acc,
      Iter = iterations,
      Runtime_sec = runtime,
      Status = "Success",
      stringsAsFactors = FALSE
    )
    
  }, error = function(e) {
    data.frame(
      Run_ID = run,
      Dimension = p,
      Center_Dist = center_dist,
      Seed = current_seed,
      Accuracy = NA,
      Iter = NA,
      Runtime_sec = NA,
      Status = paste("Error:", e$message),
      stringsAsFactors = FALSE
    )
  })
}

stopCluster(cl)

total_end_time <- Sys.time()
cat("\n\nSimulation Complete.\n")
cat("Total Runtime:", round(as.numeric(difftime(total_end_time, total_start_time, units = "mins")), 2), "minutes.\n")

if (is.null(results_log)) {
    cat("No results returned.\n")
} else {
    # 5. Output Summary
    summary_table <- results_log %>%
      filter(Status == "Success") %>%
      summarise(
        Avg_Accuracy = mean(Accuracy, na.rm = TRUE),
        SD_Accuracy = sd(Accuracy, na.rm = TRUE),
        Avg_Iter = mean(Iter, na.rm = TRUE),
        Avg_Runtime_sec = mean(Runtime_sec, na.rm = TRUE),
        Success_Count = n(),
        Total_Runs = nrow(results_log)
      )
    
    print(as.data.frame(summary_table))
    
    # Save full log (optional)
    write.csv(results_log, "benchmark_parallel_log.csv", row.names = FALSE)
    cat("Detailed log saved to benchmark_parallel_log.csv\n")
}
