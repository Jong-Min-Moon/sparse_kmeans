# ------------------------------------------------------------------
# run_simulation_identity.R
# Main driver script for evaluating methods over varying p
# ------------------------------------------------------------------
library(foreach)
library(doParallel)

# Ensure we are in the script's directory
args <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0 && file.exists(script_name)) {
    setwd(dirname(normalizePath(script_name)))
}

source("../../code_r/data_generator.R")
source("methods_wrapper.R")
source("accuracy_utils.R")

# ---------------------------------------------------------
# Centrally Defined Simulation Parameters
# ---------------------------------------------------------
n <- 200
dims <- c(6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000)
K <- 2
s <- 10
sep <- 4
n_runs <- 100 # based on typical runs

# Setup parallel backend
n_cores <- parallel::detectCores() - 1
if (n_cores < 1) n_cores <- 1

cl <- makeCluster(n_cores)
registerDoParallel(cl)

cat(sprintf("Starting Unified Simulation (Varying p) using %d cores...\n", n_cores))
dir.create("logs", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

overall_start_time <- Sys.time()

# 1. Loop Refactor: Iterate over p (dims) instead of sep
for (p in dims) {
    cat(sprintf("\n--- Starting jobs for p = %d ---\n", p))
    
    # pvalcut must be defined inside the loop now since it depends on p
    pvalcut <- log(p) / p

    # Parallel loop over independent replications
    results <- foreach(
        job_id = 1:n_runs,
        .combine = rbind,
        .packages = c("clue", "sparcl", "MASS", "methods"),
        .export = c(
            "get_specification_identity", "generate_data_from_specification", 
            "run_all_methods", "compute_all_accuracies", "run_witten",
            "run_arias", "if_pca", "get_cluster_acc", "hill_climb", "Alternate"
        )
    ) %dopar% {
        # Establish strict reproducibility per replication
        current_seed <- 2025 + job_id
        set.seed(current_seed)

        # 2. Data Generation Consistency: Generate based on varying p
        support <- 1:s
        spec <- get_specification_identity(support, sep, p)
        data_res <- generate_data_from_specification(spec, n, seed = 2026 + job_id)
        X <- data_res$X
        true_labels <- data_res$labels
        
        # Add basic dimension check for robustness
        stopifnot(nrow(X) == p)
        stopifnot(ncol(X) == n)

        # Apply ALL Methods to the SAME `X`
        methods_out <- run_all_methods(X, K, pvalcut, seed = current_seed)

        # Compute Accuracies
        acc_out <- compute_all_accuracies(methods_out, true_labels)

        # 3. Result Storage: Tidy format (long format)
        build_res <- function(method_name, list_out, acc) {
            L_val <- if(is.null(list_out$L)) NA else list_out$L
            data.frame(
                p = p,
                method = method_name,
                metric = c("accuracy", "runtime", "features_selected"),
                value = c(acc, list_out$runtime, L_val),
                replicate = job_id,
                stringsAsFactors = FALSE
            )
        }
        
        res_witten <- build_res("Witten", methods_out$witten, acc_out$acc_witten)
        res_arias  <- build_res("Arias", methods_out$arias, acc_out$acc_arias)
        res_ifpca  <- build_res("IF-PCA", methods_out$ifpca, acc_out$acc_ifpca)
        
        res_df <- rbind(res_witten, res_arias, res_ifpca)

        # Intermediate Result Logging
        n_witten <- methods_out$witten$L
        n_arias <- methods_out$arias$L
        n_ifpca <- methods_out$ifpca$L

        acc_w <- acc_out$acc_witten
        acc_a <- acc_out$acc_arias
        acc_i <- acc_out$acc_ifpca

        log_msg <- sprintf(
            "Rep %d, p = %d: Witten [feat = %s, acc = %.3f], Arias [feat = %s, acc = %.3f], IF-PCA [feat = %s, acc = %.3f]\n",
            job_id, p,
            ifelse(is.na(n_witten), "NA", as.character(n_witten)), acc_w,
            ifelse(is.na(n_arias), "NA", as.character(n_arias)), acc_a,
            ifelse(is.na(n_ifpca), "NA", as.character(n_ifpca)), acc_i
        )

        cat(log_msg)
        cat(log_msg, file = "logs/intermediate_progress.txt", append = TRUE)

        # 4. File Naming and Saving: Uses p in filename
        # Save independent file as a backup
        saveRDS(res_df, file = sprintf("results/sim_id%d_p%d.rds", job_id, p))

        return(res_df)
    }

    # Save aggregated results for this dimension (p)
    saveRDS(results, file = sprintf("results_aggregated_p%d.rds", p))
}

stopCluster(cl)

overall_runtime <- as.numeric(difftime(Sys.time(), overall_start_time, units = "mins"))
cat(sprintf("\nExecution completed in %.2f minutes.\n", overall_runtime))
