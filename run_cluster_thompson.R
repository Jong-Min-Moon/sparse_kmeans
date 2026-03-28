cat("Loading libraries...\n")
library(Matrix)
library(CVXR)
library(mclust)
library(sparcl)

# Load data
cat("Loading data...\n")
# data <- read.csv("real_data/difficult_prompts_dense_embeddings.csv", stringsAsFactors = FALSE)
data <- read.csv("real_data/difficult_prompts_sparse_embeddings.csv", stringsAsFactors = FALSE)
# Extract True_Label and Matrix
true_labels <- data$True_Label
cat("Extracting X...\n")
X <- as.matrix(data[, 3:ncol(data)])

# Subsample 50 observations randomly
cat("Subsampling 50 observations randomly without replacement...\n")
set.seed(42) # for reproducibility
sample_idx <- sample(1:nrow(X), 50, replace = FALSE)
X <- X[sample_idx, , drop = FALSE]
true_labels <- true_labels[sample_idx]

# Add independent Laplace noise
cat("Adding independent Laplace noise...\n")
set.seed(42) # Set seed for noise generation reproducibility
scale_param <- 0.0001 # Increased noise parameter multiplier to 2.0
u <- runif(length(X)) - 0.5
noise <- -scale_param * sign(u) * log(1 - 2 * abs(u))
X <- X + matrix(noise, nrow = nrow(X), ncol = ncol(X))

# Free memory since read.csv holds a lot of overhead
cat("Running garbage collection...\n")
rm(data)
gc()

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

# Run clustering
cat("Running cluster_thompson. K =", length(unique(true_labels)), "...\n")
set.seed(42)
# Reduced iterations to make the quick test feasible
res <- cluster_thompson(n_corrupted = 1, X = t(X), K = 2, C = 0.5, true_cluster = true_labels, n_iter = 4, n_perms = 3000, n_step_admm = 2000, p_val_threshold = 0.5)

cat("\n--- cluster_thompson Confusion matrix against True_Label ---\n")
print(table(Cluster = res$cluster, True_Label = true_labels))
cat(sprintf("cluster_thompson Accuracy: %.2f%%\n", 100 * get_cluster_acc(res$cluster, true_labels)))
if ("ari" %in% names(res)) cat("ARI:", res$ari, "\n")
cat("\n--- Execution Complete ---\n")



cat("\n--- Running Baselines ---\n")
p <- ncol(X)
pvalcut <- log(p) / p

cat("1. Running Witten's Sparse K-Means...\n")
witten_out <- run_witten(X, K = 2, seed = 42, return_list = TRUE)
cat("Witten Confusion matrix:\n")
print(table(Cluster = witten_out$cluster, True_Label = true_labels))
cat(sprintf("Witten Clustering Accuracy: %.2f%%\n", 100 * get_cluster_acc(witten_out$cluster, true_labels)))

cat("\n2. Running Arias-Castro Sparse K-Means...\n")
arias_out <- run_arias(X, K = 2, seed = 42, return_list = TRUE)
cat("Arias Confusion matrix:\n")
print(table(Cluster = arias_out$cluster, True_Label = true_labels))
cat(sprintf("Arias Clustering Accuracy: %.2f%%\n", 100 * get_cluster_acc(arias_out$cluster, true_labels)))

cat("\n3. Running IF-PCA...\n")
ifpca_out <- if_pca(Data = t(X), K = 2, rep = 500, nullsimu = TRUE, pvalcut = pvalcut, kmeansrep = 20, per = 1, seed = 42)
cat("IF-PCA Confusion matrix:\n")
print(table(Cluster = ifpca_out$labels, True_Label = true_labels))
cat(sprintf("IF-PCA Clustering Accuracy: %.2f%%\n", 100 * get_cluster_acc(ifpca_out$labels, true_labels)))

cat("\n4. Running SDP K-Means...\n")
t_sdp_start <- Sys.time()
G_all <- tcrossprod(X)
sdp_out <- sdp_kmeans(G_all, K = 2, max_iter = 2000)
t_sdp_end <- Sys.time()
cat(sprintf("SDP K-Means took: %.4f seconds\n", as.numeric(difftime(t_sdp_end, t_sdp_start, units = "secs"))))
cat("SDP K-Means Confusion matrix:\n")
print(table(Cluster = sdp_out$cluster, True_Label = true_labels))
cat(sprintf("SDP K-Means Clustering Accuracy: %.2f%%\n", 100 * get_cluster_acc(sdp_out$cluster, true_labels)))
