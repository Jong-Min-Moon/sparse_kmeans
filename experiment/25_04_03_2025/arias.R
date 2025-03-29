install.packages("phyclust")
library(sparcl)
library(RSQLite)
source("/home1/jongminm/sparse_kmeans/experiment/25_04_03_2025/competitor.R")
source("/home1/jongminm/sparse_kmeans/experiment/25_04_03_2025/data_generation.R")

# Params
sep <- 4
p_seq <- c(50, seq(500, 5000, by = 500) )  # or use seq(50, 5000, by = 450) to match your list
n <- 200
s <- 10
n_rep <- 200

# SQLite setup
db_dir <- "/home1/jongminm/sparse_kmeans/sparse_kmeans.db"
db <- dbConnect(SQLite(), db_dir)



# Simulation loop
for (p in p_seq) {
  for (rep in 1:n_rep) {
    seed <- 10000 + p + rep  # reproducible
    data <- generate_isotropic_Gaussian(sep, s, p, n, seed)
    x <- data$x
    cluster_true <- data$cluster_true
    
    cluster_est <- tryCatch({
      sparse_kmeans_hillclimb(x)
    }, error = function(e) {
      rep(NA, n)
    })
    
    acc <- if (all(is.na(cluster_est))) 0 else evaluate_clustering(cluster_est, cluster_true)
    
    dbExecute(db, "
      INSERT INTO iso_arias (
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
