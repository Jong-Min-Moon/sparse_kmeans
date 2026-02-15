# 1. Load Required Packages
# Set mirror for non-interactive installation
local({r <- getOption("repos")
       r["CRAN"] <- "https://cloud.r-project.org" 
       options(repos=r)
})

if (!require("CVXR")) install.packages("CVXR")
if (!require("MASS")) install.packages("MASS")
if (!require("dplyr")) install.packages("dplyr") # For easier aggregation

library(CVXR)
library(MASS)
library(dplyr)

# 2. Simulation Parameters
p_values <- seq(1000, 5000, by = 1000) # Dimensions: 1000, 2000, ..., 5000
n_runs <- 20                           # Runs per dimension
N <- 200                               # Total observations
n1 <- 100; n2 <- 100                   # Cluster sizes
k <- 2                                 # Number of clusters
center_dist <- 4                       # L2 separation (Updated to 4)

# Initialize data frame to store results (Added Runtime_sec)
results_log <- data.frame(
  Dimension = integer(),
  Seed = integer(),
  Accuracy = double(),
  Status = character(),
  Runtime_sec = double(), 
  stringsAsFactors = FALSE
)

# 3. Simulation Loop
cat(sprintf("Starting Simulation... (Total runs: %d)\n", length(p_values) * n_runs))
total_start_time <- Sys.time()

for (p in p_values) {
  
  # Define Sparse Cluster Centers (Fixed for this p)
  mu1 <- rep(0, p)
  
  # mu2 is sparse: 10 non-zero entries
  nonzero_count <- 10
  val_per_entry <- sqrt(center_dist^2 / nonzero_count)
  mu2 <- rep(0, p)
  mu2[1:nonzero_count] <- val_per_entry
  
  cat(sprintf("\n========================================\n"))
  cat(sprintf("Processing Dimension p = %d\n", p))
  cat(sprintf("========================================\n"))
  
  for (run in 1:n_runs) {
    run_start_time <- Sys.time()
    
    # Unique seed for each run
    current_seed <- p + run
    set.seed(current_seed)
    
    # --- A. Data Generation ---
    X1 <- matrix(rnorm(n1 * p) + mu1, nrow = n1, byrow = TRUE)
    X2 <- matrix(rnorm(n2 * p) + mu2, nrow = n2, byrow = TRUE)
    data_matrix <- rbind(X1, X2)
    true_labels <- c(rep(1, n1), rep(2, n2))
    
    # --- B. SDP Setup ---
    # R's dist() computes distance between ROWS.
    D <- as.matrix(dist(data_matrix))^2
    Z <- Variable(N, N, PSD = TRUE)
    objective <- Minimize(sum_entries(D * Z))
    constraints <- list(
      Z >= 0,
      sum_entries(Z, axis = 1) == 1,
      matrix_trace(Z) == k
    )
    prob <- Problem(objective, constraints)
    
    # --- C. Solve (Quietly) ---
    result <- solve(prob, solver = "SCS", verbose = FALSE, max_iters = 2500)
    
    # --- D. Rounding & Accuracy ---
    if (result$status %in% c("optimal", "optimal_inaccurate")) {
      Z_sol <- result$getValue(Z)
      
      eigen_decomp <- eigen(Z_sol, symmetric = TRUE)
      V <- eigen_decomp$vectors[, 1:k]
      
      set.seed(42) 
      final_clustering <- kmeans(V, centers = k, nstart = 10)
      estimated_labels <- final_clustering$cluster
      
      conf_matrix <- table(true_labels, estimated_labels)
      acc_match <- sum(diag(conf_matrix))
      acc_swap <- sum(conf_matrix) - sum(diag(conf_matrix))
      
      accuracy <- max(acc_match, acc_swap) / N
    } else {
      accuracy <- NA
    }
    
    # --- E. Timing and Logging ---
    run_end_time <- Sys.time()
    # Convert difftime to strictly seconds
    run_duration <- as.numeric(difftime(run_end_time, run_start_time, units = "secs"))
    
    results_log <- rbind(results_log, data.frame(
      Dimension = p,
      Seed = current_seed,
      Accuracy = accuracy,
      Status = result$status,
      Runtime_sec = run_duration
    ))
    
    # Print per-run metrics mirroring Python
    acc_str <- ifelse(is.na(accuracy), "   N/A", sprintf("%5.1f%%", accuracy * 100))
    cat(sprintf("Run %02d/%02d | Seed: %d | Time: %5.2fs | Acc: %s | Status: %s\n",
                run, n_runs, current_seed, run_duration, acc_str, result$status))
  }
}

total_end_time <- Sys.time()
cat("\n\nSimulation Complete.\n")
cat("Total Runtime:", round(as.numeric(difftime(total_end_time, total_start_time, units = "mins")), 2), "minutes.\n")

# 4. Aggregation and Output
summary_table <- results_log %>%
  group_by(Dimension) %>%
  summarise(
    Avg_Accuracy = mean(Accuracy, na.rm = TRUE),
    SD_Accuracy = sd(Accuracy, na.rm = TRUE),
    Successful_Runs = sum(!is.na(Accuracy)),
    Avg_Runtime_sec = mean(Runtime_sec, na.rm = TRUE)
  )

print(as.data.frame(summary_table))
