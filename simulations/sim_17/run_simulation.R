# ------------------------------------------------------------------
# run_simulation.R
# Main driver script for sim_17 (Unified Sparse Clustering Evaluation)
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
p <- 400
K <- 2
rho <- 0.45
precision_sparsity <- 2
support <- 1:10
flip <- FALSE
separations <- c(8,9,10,11)
n_runs <- 20

# IF-PCA specific
pvalcut <- log(p) / p

# Setup parallel backend
n_cores <- parallel::detectCores() - 1
if (n_cores < 1) n_cores <- 1

cl <- makeCluster(n_cores)
registerDoParallel(cl)

cat(sprintf("Starting Unified Simulation (sim_17) using %d cores...\n", n_cores))
dir.create("logs", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

overall_start_time <- Sys.time()

for (sep in separations) {
    cat(sprintf("\n--- Starting jobs for separation = %d ---\n", sep))

    # Initialize Generator once per separation
    generator <- get_specification_chaingraph(
        support = support,
        separation = sep,
        dimension = p,
        precision_sparsity = precision_sparsity,
        conditional_correlation = rho,
        flip = flip
    )

    # Parallel loop over independent replications
    # Note: We must export loaded functions properly in foreach to avoid worker failing
    results <- foreach(
        job_id = 1:n_runs,
        .combine = rbind,
        .packages = c("clue", "sparcl", "MASS", "methods"),
        .export = c(
            "get_specification_chaingraph", "generate_data_from_specification",
            "run_all_methods", "compute_all_accuracies", "run_witten",
            "run_arias", "if_pca", "get_cluster_acc", "hill_climb", "Alternate"
        )
    ) %dopar% {
        # 1. Establish strict reproducibility per replication
        current_seed <- 2025 + job_id
        set.seed(current_seed)

        # 2. Generate Data Exactly Once Per Replication
        data_res <- generate_data_from_specification(generator, n, seed = current_seed)
        X <- data_res$X


        true_labels <- data_res$labels

        # 3. Apply ALL Methods to the SAME `X`
        methods_out <- run_all_methods(X, K, pvalcut, seed = current_seed)

        # 4. Compute Accuracies
        acc_out <- compute_all_accuracies(methods_out, true_labels)

        # 5. Store / Return Structured Result
        res_df <- data.frame(
            job_id = job_id,
            p = p,
            n = n,
            sep = sep,
            rho = rho,
            accuracy_witten = acc_out$acc_witten,
            runtime_witten = methods_out$witten$runtime,
            accuracy_arias = acc_out$acc_arias,
            runtime_arias = methods_out$arias$runtime,
            accuracy_ifpca = acc_out$acc_ifpca,
            ifpca_L = methods_out$ifpca$L,
            runtime_ifpca = methods_out$ifpca$runtime
        )

        # 6. Intermediate Result Logging
        n_witten <- methods_out$witten$L
        n_arias <- methods_out$arias$L
        n_ifpca <- methods_out$ifpca$L

        acc_w <- acc_out$acc_witten
        acc_a <- acc_out$acc_arias
        acc_i <- acc_out$acc_ifpca

        log_msg <- sprintf(
            "Rep %d, sep = %d: Witten [feat = %s, acc = %.3f], Arias [feat = %s, acc = %.3f], IF-PCA [feat = %s, acc = %.3f]\n",
            job_id, sep,
            ifelse(is.na(n_witten), "NA", as.character(n_witten)), acc_w,
            ifelse(is.na(n_arias), "NA", as.character(n_arias)), acc_a,
            ifelse(is.na(n_ifpca), "NA", as.character(n_ifpca)), acc_i
        )

        cat(log_msg)
        cat(log_msg, file = "logs/intermediate_progress.txt", append = TRUE)

        # Save independent file as a backup
        saveRDS(res_df, file = sprintf("results/sim_id%d_sep%d.rds", job_id, sep))

        return(res_df)
    }
}

stopCluster(cl)

overall_runtime <- as.numeric(difftime(Sys.time(), overall_start_time, units = "mins"))
cat(sprintf("\nExecution completed in %.2f minutes.\n", overall_runtime))
cat("You can now run 'Rscript aggregate_results.R' to view metrics.\n")
