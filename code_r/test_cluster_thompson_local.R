library(mclust)
library(MASS)
library(cluster)

source("d:/GitHub/sparse_kmeans/code_r/utils.R")
source("d:/GitHub/sparse_kmeans/code_r/sdp_kmeans.R")
source("d:/GitHub/sparse_kmeans/code_r/clustering_block_knowncov.R")
source("d:/GitHub/sparse_kmeans/code_r/reward_thompson.R")
source("d:/GitHub/sparse_kmeans/code_r/cluster_thompson.R")
source("d:/GitHub/sparse_kmeans/code_r/data_generator.R")
source("d:/GitHub/sparse_kmeans/code_r/get_cluster_acc.R")

# Simulation Parameters Configuration (Small scale)
n <- 200
K <- 2
p <- 500
support <- 1:10
separation <- 4
pval <- 0.01

set.seed(42)

# Data Generation Process
cat("Generating data...\n")
generator_spec <- get_specification_identity(
  support = support,
  separation = separation,
  dimension = p
)

data_res <- generate_data_from_specification(generator_spec, n, seed = 42)
X <- data_res$X
true_labels <- data_res$labels

# Experimental Clustering Evaluation
cat("Launching `cluster_thompson` evaluation...\n")
best_res <- NULL
best_sil <- -Inf
best_C <- NA

c_values <- c(0.5, 0.4)

for (c_val in c_values) {
  cat(sprintf("\n--- Evaluating C = %.1f ---\n", c_val))
  res_temp <- cluster_thompson(
    X = X,
    K = K,
    n_iter = 5,
    C = c_val,
    n_perms = 500,
    p_val_threshold = pval,
    n_step_admm = 500,
    covariance = NULL, # Driver passes NULL
    true_cluster = true_labels
  )
  
  if (!is.null(res_temp)) {
    selected_indices <- which(res_temp$selected)
    n_sel <- length(selected_indices)
    
    if (n_sel > 0 && length(unique(res_temp$cluster)) > 1) {
      dist_mat <- dist(t(X[selected_indices, , drop = FALSE]))
      sil <- cluster::silhouette(res_temp$cluster, dist_mat)
      avg_sil <- mean(sil[, 3])
    } else {
      avg_sil <- -1
    }
    
    cat(sprintf(">>> C = %.1f completed. Silhouette Index: %.4f (Selected Features: %d)\n", c_val, avg_sil, n_sel))
    
    if (avg_sil > best_sil) {
      best_sil <- avg_sil
      best_res <- res_temp
      best_C <- c_val
    }
  }
}

res <- best_res
cat(sprintf("\n=== Selected Optimal C = %.1f (Silhouette Index: %.4f) ===\n", best_C, best_sil))

if (!is.null(res)) {
  acc <- get_cluster_acc(res$cluster, true_labels)
  selected_indices <- which(res$selected)
  tp <- length(intersect(selected_indices, support))
  
  cat(sprintf("\nFinal Accuracy: %.4f\n", acc))
  cat(sprintf("True Positives: %d / %d\n", tp, length(support)))
  cat(sprintf("Total Selected: %d\n", length(selected_indices)))
}
