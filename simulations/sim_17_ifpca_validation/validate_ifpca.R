# ---------------------------------------------------------
# Validation Script for IF-PCA parameters against MATLAB
# ---------------------------------------------------------
library(foreach)
library(doParallel)

# Ensure we are in the script's directory
args <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0 && file.exists(script_name)) {
    setwd(dirname(normalizePath(script_name)))
}

source("../../code_r/sparse_symmetric_data_generator.R")
source("../../code_r/ifpca.R")
source("../../code_r/get_cluster_acc.R")

# ---------------------------------------------------------
# Exact MATLAB Configuration
# ---------------------------------------------------------
n <- 500
p <- 100
separation <- 4
rho <- 0.2
K <- 2
precision_sparsity <- 2
support <- 1:10
flip <- FALSE
n_runs <- 100

cat("--------------------------------------------------\n")
cat("Starting IF-PCA MATLAB Validation Test\n")
cat(sprintf("Params: n = %d, p = %d, separation = %.1f, rho = %.2f\n", n, p, separation, rho))
cat(sprintf("Iterations: %d\n", n_runs))
cat("--------------------------------------------------\n")

# Setting up Parallel processing
n_cores <- parallel::detectCores() - 1
if (n_cores < 1) n_cores <- 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)

generator <- sparse_symmetric_data_generator(
    support = support,
    separation = separation,
    dimension = p,
    precision_sparsity = precision_sparsity,
    conditional_correlation = rho,
    flip = flip
)

start_time <- Sys.time()

results <- foreach(job_id = 1:n_runs, .combine = rbind, .packages = c("clue"), .export = c("sparse_symmetric_data_generator", "generate_data_from_generator", "if_pca", "get_cluster_acc")) %dopar% {
    # Set isolated seed for reproducibility
    current_seed <- 2025 + job_id
    set.seed(current_seed)

    # Generate Data
    data_res <- generate_data_from_generator(generator, n, seed = current_seed)
    X <- data_res$X
    true_labels <- data_res$labels

    # data_res$X comes out as n x p.
    # ifpca.R expects features in rows (p x n).
    if (nrow(X) == n && ncol(X) == p) {
        X_ifpca <- t(X)
    } else {
        X_ifpca <- X
    }

    st_algo <- Sys.time()
    ifpca_res <- tryCatch(
        {
            if_pca(Data = X_ifpca, K = K, rep = 500, nullsimu = TRUE, pvalcut = log(p) / p, kmeansrep = 20, per = 1, seed = current_seed)
        },
        error = function(e) {
            return(NULL)
        }
    )

    if (!is.null(ifpca_res)) {
        acc <- get_cluster_acc(ifpca_res$labels, true_labels)
        L <- ifpca_res$L
    } else {
        acc <- NA
        L <- NA
    }

    data.frame(
        job_id = job_id,
        acc = acc,
        L = L
    )
}

stopCluster(cl)
runtime <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

# ---------------------------------------------------------
# Summary outputs
# ---------------------------------------------------------

mean_acc <- mean(results$acc, na.rm = TRUE)
sd_acc <- sd(results$acc, na.rm = TRUE)
min_acc <- min(results$acc, na.rm = TRUE)
max_acc <- max(results$acc, na.rm = TRUE)
mean_L <- mean(results$L, na.rm = TRUE)

cat(sprintf("\n--- FINAL RESULTS ---\n"))
cat(sprintf("Execution Time: %.2f mins\n", runtime))
cat(sprintf("Mean Accuracy: %.4f (Target: ~0.7000)\n", mean_acc))
cat(sprintf("SD Accuracy:   %.4f\n", sd_acc))
cat(sprintf("Min Accuracy:  %.4f\n", min_acc))
cat(sprintf("Max Accuracy:  %.4f\n", max_acc))
cat(sprintf("Mean Features (L): %.2f\n", mean_L))

# Write log to disk
write.csv(results, "validation_results.csv", row.names = FALSE)
cat("\nResults saved to validation_results.csv\n")
