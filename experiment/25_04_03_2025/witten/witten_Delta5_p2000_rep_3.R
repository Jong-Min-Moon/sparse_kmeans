rep_start = 401
rep_end = 600
dimension  = 2000
separation = 5
library(sparcl)
library(RSQLite)
source("/home1/jongminm/sparse_kmeans/experiment/25_04_03_2025/competitor.R")
source("/home1/jongminm/sparse_kmeans/experiment/25_04_03_2025/data_generation.R")

# Params

n <- 200
s <- 10
n_rep = rep_end-rep_start+1
print(dimension)
print(separation)
# SQLite setup
db_dir <- "/home1/jongminm/sparse_kmeans/sparse_kmeans.db"
db <- dbConnect(SQLite(), db_dir)

# Safe insert with retry logic
safe_db_insert <- function(db, query, params) {
  repeat {
    result <- tryCatch({
      dbExecute(db, query, params = params)
      TRUE
    }, error = function(e) {
      if (grepl("database is locked", conditionMessage(e))) {
        cat("Database is locked. Retrying in 5 seconds...\n")
        Sys.sleep(5)
        FALSE
      } else {
        stop(e)
      }
    })
    if (result) break
  }
}

# SQL query (same for all inserts)
insert_query <- "
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
"

# Simulation loop
  results_list <- list()

  for (rep in rep_start:rep_end) {
    seed <- 10000 + dimension + rep
    data <- generate_isotropic_Gaussian(separation, s, dimension, n, seed)
    x <- data$x
    cluster_true <- data$cluster_true

    cluster_est <- tryCatch({
      sparse_kmeans_ell1_ell2(x)
    }, error = function(e) {
      rep(NA, n)
    })

    acc <- if (all(is.na(cluster_est))) 0 else evaluate_clustering(cluster_est, cluster_true)

    results_list[[rep]] <- list(
      rep = rep, sep = separation, dim = dimension, acc = acc
    )

    cat(sprintf("Finished: p = %d, rep = %d, acc = %.3f\n", dimension, rep, acc))
  }

  for (res in results_list) {
    safe_db_insert(db, insert_query, res)
  }

  cat(sprintf("Saved to DB for p = %d after %d reps\n", dimension, n_rep))

dbDisconnect(db)
