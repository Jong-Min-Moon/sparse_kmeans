# ------------------------------------------------------------------
# sim_gaussian.R
# Standalone HPC Array Script: Competitor Evaluation under Gaussian Noise
# ------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
job_id <- 1
p <- 100
sep <- 6

if (length(args) > 0) {
    for (i in seq_along(args)) {
        if (args[i] == "--job_id" && i < length(args)) job_id <- as.integer(args[i+1])
        if (args[i] == "--p"      && i < length(args)) p      <- as.integer(args[i+1])
        if (args[i] == "--sep"    && i < length(args)) sep    <- as.numeric(args[i+1])
    }
}

noise <- "Gaussian"
n <- 200
rho <- 0.45
precision_sparsity <- 2

cat(sprintf("\n--- HPC Driver (Erdos-Renyi): Gaussian Job %d, p=%d, sep=%.1f ---\n", job_id, p, sep))

# 1. Source Dependencies directly from code_r
source("../../../code_r/data_generator.R")
source("../../../code_r/competitors_modernized.R")
source("../../../code_r/ifpca.R")
source("../../../code_r/get_cluster_acc.R")

# 2. Data Generation
set.seed(2025 + job_id * 1000 + p)
spec <- get_specification_erdos_renyi(
    p = p,
    separation = sep,
    s = 10
)
data_res <- generate_data_from_specification(
    specification = spec,
    n = n,
    seed = 2025 + job_id * 1000 + p,
    noise = noise
)

X <- t(data_res$X)
true_labels <- data_res$labels

# 3. Helpers
.try_method <- function(expr, name, n) {
    tryCatch(expr, error = function(e) {
        warning(sprintf("%s failed: %s", name, e$message))
        list(cluster = rep(NA, n), L = NA)
    })
}

.safe_acc <- function(cluster, true_labels) {
    if (!any(is.na(cluster))) get_cluster_acc(cluster, true_labels) else NA
}

# 4. Execute Methods
pvalcut <- log(p) / p
current_seed <- 2025 + job_id * 1000 + p

# Witten
st <- Sys.time()
witten_res <- .try_method(
    run_witten(X, K = 2, seed = current_seed, return_list = TRUE),
    "Witten", n
)
rt_witten <- as.numeric(difftime(Sys.time(), st, units = "secs"))

# Arias
st <- Sys.time()
arias_res <- .try_method(
    run_arias(X, K = 2, seed = current_seed, return_list = TRUE),
    "Arias", n
)
rt_arias <- as.numeric(difftime(Sys.time(), st, units = "secs"))

# IF-PCA
X_ifpca <- t(X)
st <- Sys.time()
ifpca_res <- tryCatch(
    if_pca(
        Data = X_ifpca, K = 2, rep = 500, nullsimu = TRUE,
        pvalcut = pvalcut, kmeansrep = 20, per = 1, seed = current_seed
    ),
    error = function(e) {
        warning(paste("IF-PCA failed:", e$message))
        NULL
    }
)
rt_ifpca <- as.numeric(difftime(Sys.time(), st, units = "secs"))

# 5. Output
acc_witten <- .safe_acc(witten_res$cluster, true_labels)
acc_arias  <- .safe_acc(arias_res$cluster, true_labels)
acc_ifpca  <- .safe_acc(if (!is.null(ifpca_res)) ifpca_res$labels else rep(NA, n), true_labels)

res_df <- data.frame(
    job_id          = job_id,
    p               = p,
    n               = n,
    sep             = sep,
    rho             = rho,
    accuracy_witten = acc_witten,
    runtime_witten  = rt_witten,
    accuracy_arias  = acc_arias,
    runtime_arias   = rt_arias,
    accuracy_ifpca  = acc_ifpca,
    ifpca_L         = if (!is.null(ifpca_res)) as.numeric(ifpca_res$L) else NA,
    runtime_ifpca   = rt_ifpca
)

log_msg <- sprintf(
    "[%s] Rep %d, p = %d: Witten [feat=%s, acc=%.3f], Arias [feat=%s, acc=%.3f], IF-PCA [feat=%s, acc=%.3f]\n",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"), job_id, p,
    ifelse(is.na(witten_res$L), "NA", as.character(witten_res$L)), acc_witten,
    ifelse(is.na(arias_res$L), "NA", as.character(arias_res$L)), acc_arias,
    ifelse(is.null(ifpca_res) || is.na(ifpca_res$L), "NA", as.character(ifpca_res$L)), acc_ifpca
)

# 6. Save Checkpoint
out_dir <- file.path("results_raw", tolower(noise), sprintf("p%d", p))
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

out_file <- file.path(out_dir, sprintf("sim_job%d_p%d.rds", job_id, p))
saveRDS(res_df, file = out_file)

cat(log_msg)
cat(sprintf("Result saved to %s\n", out_file))
