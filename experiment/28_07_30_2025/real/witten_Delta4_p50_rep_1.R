rep_start = 1
rep_end = 200
dimension  = 50
separation = 4
library(sparcl)
library(RSQLite)
source("/home1/jongminm/sparse_kmeans/experiment/25_04_03_2025/competitor.R")
source("/home1/jongminm/sparse_kmeans/experiment/25_04_03_2025/data_generation.R")

x = 2^lx.original 
y = ly.original
      cluster_est = sparse_kmeans_ell1_ell2(x[20:65])
    evaluate_clustering(cluster_est, ly.original[20:65])

    cluster_arias = sparse_kmeans_hillclimb(x[30:65,])
    evaluate_clustering(cluster_arias, ly.original[30:65] )
    
    acc_witten = 0
    acc_arias = 0
    for (i in 1:30){
      data = get_proportional_subsample(x, y, 45, i)  
      sol_witten = sparse_kmeans_ell1_ell2(data$X)
      acc_witten = acc_witten + evaluate_clustering(sol_witten, data$y)/30
      
      sol_arias = sparse_kmeans_hillclimb(data$X)
      acc_arias = acc_arias + evaluate_clustering(sol_arias, data$y)/30
    }
    
    
    
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
    