cat("Loading libraries...\n")
library(Matrix)
library(CVXR)
library(mclust)
library(sparcl)

# Source dependencies
cat("Sourcing dependencies...\n")
source("code_r/utils.R")
source("code_r/clustering_block_knowncov.R")
source("code_r/sdp_kmeans.R")
source("code_r/get_cluster_acc.R")
source("code_r/reward_thompson.R")
source("code_r/cluster_thompson.R")

source("code_r/competitors_modernized.R")
source("code_r/ifpca.R")

# Load data
cat("Loading data...\n")
data <- read.csv("real_data/difficult_prompts_dense_embeddings.csv", stringsAsFactors = FALSE)

# Extract True_Label and Matrix
true_labels_full <- data$True_Label
cat("Extracting X...\n")
X_full <- as.matrix(data[, 3:ncol(data)])

# Free memory
rm(data)
gc()

num_seeds <- 100
acc_thompson <- numeric(num_seeds)
acc_witten <- numeric(num_seeds)
acc_arias <- numeric(num_seeds)
acc_ifpca <- numeric(num_seeds)
acc_sdp <- numeric(num_seeds)

for (i in 1:num_seeds) {
    cat(sprintf("\n--- Starting Iteration %d/%d (Seed %d) ---\n", i, num_seeds, i))

    set.seed(i)

    # Subsample 50 observations randomly
    sample_idx <- sample(1:nrow(X_full), 50, replace = FALSE)
    X <- X_full[sample_idx, , drop = FALSE]
    true_labels <- true_labels_full[sample_idx]

    # Add independent Laplace noise
    scale_param <- 0.07
    u <- runif(length(X)) - 0.5
    noise <- -scale_param * sign(u) * log(1 - 2 * abs(u))
    X <- X + matrix(noise, nrow = nrow(X), ncol = ncol(X))

    p <- ncol(X)
    pvalcut <- log(p) / p

    # 1. cluster_thompson
    cat("Running cluster_thompson...\n")
    res <- cluster_thompson(n_corrupted = 1, X = t(X), K = 2, C = 0.5, true_cluster = true_labels, n_iter = 50, n_perms = 3000, n_step_admm = 2000, p_val_threshold = 0.5)
    acc_thompson[i] <- get_cluster_acc(res$cluster, true_labels)

    # 2. Witten
    cat("Running Witten's Sparse K-Means...\n")
    witten_out <- run_witten(X, K = 2, seed = i, return_list = TRUE)
    acc_witten[i] <- get_cluster_acc(witten_out$cluster, true_labels)
    print(acc_witten[i])
    # 3. Arias
    cat("Running Arias-Castro Sparse K-Means...\n")
    arias_out <- run_arias(X, K = 2, seed = i, return_list = TRUE)
    acc_arias[i] <- get_cluster_acc(arias_out$cluster, true_labels)
    print(acc_arias[i])
    # 4. IF-PCA
    cat("Running IF-PCA...\n")
    ifpca_out <- if_pca(Data = t(X), K = 2, rep = 500, nullsimu = TRUE, pvalcut = pvalcut, kmeansrep = 20, per = 1, seed = i)
    acc_ifpca[i] <- get_cluster_acc(ifpca_out$labels, true_labels)
    print(acc_ifpca[i])
}

cat("\n=========================================\n")
cat("Final Average Accuracies over", num_seeds, "seeds:\n")
cat(sprintf("cluster_thompson: %.2f%%\n", 100 * mean(acc_thompson)))
cat(sprintf("Witten:           %.2f%%\n", 100 * mean(acc_witten)))
cat(sprintf("Arias-Castro:     %.2f%%\n", 100 * mean(acc_arias)))
cat(sprintf("IF-PCA:           %.2f%%\n", 100 * mean(acc_ifpca)))
cat("=========================================\n")
