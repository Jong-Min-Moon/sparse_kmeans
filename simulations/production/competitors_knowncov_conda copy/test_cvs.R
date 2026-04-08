# ------------------------------------------------------------------
# test_cvs.R
# Smoke test: clustvarsel integration into run_simulation_methods.
# Uses small p=50 to keep the test fast.
# ------------------------------------------------------------------
setwd(dirname(normalizePath(if (interactive()) getwd() else {
    args <- commandArgs(trailingOnly = FALSE)
    sub("--file=", "", args[grep("--file=", args)])
})))

source("sim_utils.R")

n <- 200; p <- 3000; sep <- 4; n_reps <- 10
cat(sprintf("R %s | p=%d n=%d | reps=%d\n", getRversion(), p, n, n_reps))

methods_to_run <- c("arias")
results_list <- list()

for (i in 1:n_reps) {
    seed <- 42 + i
    cat(sprintf("Rep %d (seed=%d)... ", i, seed))
    
    # Generate data
    data_res <- generate_data_knowncov(n = n, p = p, sep = sep,
                                       seed = seed, noise_type = "Laplace")
    
    st <- Sys.time()
    sim_out <- run_simulation_methods(
        X           = data_res$X,
        true_labels = data_res$true_labels,
        K           = 2,
        p           = p,
        n           = n,
        sep         = sep,
        rho         = 0,
        job_id      = i,
        seed        = seed,
        methods     = methods_to_run
    )
    et <- Sys.time()
    
    acc <- sim_out$res_df$accuracy_arias
    rt <- as.numeric(difftime(et, st, units = "secs"))
    L <- sim_out$res_df$arias_L # Wait, check column name
    # Actually res_df columns for arias are accuracy_arias, runtime_arias
    # Let's check the code_r/methods_wrapper.R for available columns.
    
    results_list[[i]] <- sim_out$res_df
    cat(sprintf("Acc: %.3f | Time: %.1fs\n", acc, rt))
}

final_df <- do.call(rbind, results_list)
cat("\n--- Final Results (Averages over", n_reps, "reps) ---\n")
cat(sprintf("Average Accuracy: %.4f (sd: %.4f)\n", 
            mean(final_df$accuracy_arias), sd(final_df$accuracy_arias)))
cat(sprintf("Average Runtime:  %.2f s\n", mean(final_df$runtime_arias)))
cat("--- DONE ---\n")
