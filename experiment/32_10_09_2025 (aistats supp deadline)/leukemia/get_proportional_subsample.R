get_proportional_subsample <- function(X, y, subsample_size, seed_val,
                                       cluster1_label = 1, cluster2_label = 2) {
  
  set.seed(seed_val) # Set the random seed for reproducibility
  
  # Calculate the proportion of cluster 1 among the two specified clusters
  num_cluster_1_total <- sum(y == cluster1_label)
  num_cluster_2_total <- sum(y == cluster2_label)
  total_two_clusters <- num_cluster_1_total + num_cluster_2_total
  
  percent_cluster_1 <- 0.5 # Default if no data for these clusters
  if (total_two_clusters > 0) {
    percent_cluster_1 <- num_cluster_1_total / total_two_clusters
  } else {
    warning(paste0("No data points found for cluster '", cluster1_label,
                   "' or '", cluster2_label, "'. Defaulting percent_cluster_1 to 0.5."))
  }
  
  # Identify indices for each cluster
  idx_cluster_1 <- which(y == cluster1_label)
  idx_cluster_2 <- which(y == cluster2_label)
  
  # Calculate desired subsample sizes for each cluster
  desired_subsample_size_cluster_1 <- floor(subsample_size * percent_cluster_1)
  desired_subsample_size_cluster_2 <- subsample_size - desired_subsample_size_cluster_1
  
  # --- Adjust for available samples (Crucial for robustness) ---
  actual_subsample_size_cluster_1 <- min(desired_subsample_size_cluster_1, length(idx_cluster_1))
  actual_subsample_size_cluster_2 <- min(desired_subsample_size_cluster_2, length(idx_cluster_2))
  
  if (actual_subsample_size_cluster_1 + actual_subsample_size_cluster_2 < subsample_size) {
    warning(paste0("Cannot achieve exact desired subsample_size (", subsample_size, ") ",
                   "with original proportions due to limited samples in one or both clusters. ",
                   "Actual subsample sizes: Cluster '", cluster1_label, "' = ", actual_subsample_size_cluster_1,
                   ", Cluster '", cluster2_label, "' = ", actual_subsample_size_cluster_2,
                   ". Total actual subsample size: ", actual_subsample_size_cluster_1 + actual_subsample_size_cluster_2))
  }
  
  # Select samples from cluster 1
  perm_idx_cluster_1 <- sample(length(idx_cluster_1))
  selected_idx_cluster_1 <- idx_cluster_1[perm_idx_cluster_1[1:actual_subsample_size_cluster_1]]
  
  # Select samples from cluster 2
  perm_idx_cluster_2 <- sample(length(idx_cluster_2))
  selected_idx_cluster_2 <- idx_cluster_2[perm_idx_cluster_2[1:actual_subsample_size_cluster_2]]
  
  # Combine the selected indices (order of classes preserved)
  final_idx <- c(selected_idx_cluster_1, selected_idx_cluster_2)
  
  # Extract the subsampled data and labels
  # Assuming X is (features x data_points), so we select columns
  X_new <- X[final_idx, , drop = FALSE] # drop=FALSE ensures matrix if only 1 column
  y_new <- y[final_idx]
  
  return(list(X = X_new, y = y_new))
}