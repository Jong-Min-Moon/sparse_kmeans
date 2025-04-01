library(sparcl)
library(RSQLite)

# Params
sep <- 5
p_seq <- c(50, seq(500, 5000, by = 500) )  # or use seq(50, 5000, by = 450) to match your list
n <- 200
s <- 10
n_rep <- 200

# SQLite setup
db_dir <- "/home1/jongminm/sparse_kmeans/sparse_kmeans.db"
db <- dbConnect(SQLite(), db_dir)



# Functions
generate_isotropic_Gaussian <- function(sep, s, p, n, seed) {
  K <- 2
  cluster_true <- c(rep(1, n/2), rep(2, n/2))
  M <- 0.5 * sep / sqrt(s)
  sparse_mean <- c(rep(1, s), rep(0, p - s))
  mu_1 <- -M * sparse_mean
  mu_2 <-  M * sparse_mean
  x_noiseless <- t(sapply(cluster_true, function(cl) if (cl == 1) mu_1 else mu_2))
  
  set.seed(seed)
  x <- x_noiseless + matrix(rnorm(n * p), nrow = n, ncol = p)
  list(x = x, cluster_true = cluster_true)
}

sparse_kmeans_ell1_ell2 <- function(x) {
  x_scaled <- scale(x, TRUE, TRUE)
  km.perm <- KMeansSparseCluster.permute(x_scaled, K = 2, wbounds = seq(3, 7, length.out = 15), nperms = 5)
  km.out <- KMeansSparseCluster(x_scaled, K = 2, wbounds = km.perm$bestw)
  return(km.out[[1]]$Cs)
}

evaluate_clustering <- function(predicted, truth) {
  acc1 <- mean(predicted == truth)
  acc2 <- mean(predicted != truth)
  return(max(acc1, acc2))
}

# Simulation loop
for (p in p_seq) {
  for (rep in 1:n_rep) {
    seed <- 10000 + p + rep  # reproducible
    data <- generate_isotropic_Gaussian(sep, s, p, n, seed)
    x <- data$x
    cluster_true <- data$cluster_true
    
    cluster_est <- tryCatch({
      sparse_kmeans_ell1_ell2(x)
    }, error = function(e) {
      rep(NA, n)
    })
    
    acc <- if (all(is.na(cluster_est))) 0 else evaluate_clustering(cluster_est, cluster_true)
    
    dbExecute(db, "
      INSERT INTO iso_witten (
        rep, iter, sep, dim, rho, sparsity, stop_og, stop_sdp, stop_loop, acc,
        obj_prim, obj_dual, obj_original, true_pos, false_pos, false_neg,
        diff_x_tilde_fro, diff_x_tilde_op, diff_x_tilde_ellone,
        time_est, time_sdp, jobdate, survived_indices, cluster_est
      ) VALUES (
        :rep, 0, :sep, :dim, 0, 0, 0, 0, 0, :acc,
        0, 0, 0, 0, 0, 0,
        0, 0, 0,
        0, 0, CURRENT_TIMESTAMP, '', ''
      )
    ", params = list(rep = rep, sep = sep, dim = p, acc = acc))
    
    cat(sprintf("Finished: p = %d, rep = %d, acc = %.3f\n", p, rep, acc))
  }
}

dbDisconnect(db)
