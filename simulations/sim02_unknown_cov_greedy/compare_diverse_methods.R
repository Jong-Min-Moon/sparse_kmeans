# Comprehensive Benchmark: Diverse Clustering Algorithms
library(MASS)
library(mclust)
library(kernlab)
library(dbscan)
library(stats)

# Simulation Parameters
p <- 400
n <- 500
K <- 2
rho <- 0.45
n_reps <- 10

# Generate Covariance
Omega <- matrix(0, p, p)
for (i in 1:p) {
    Omega[i, i] <- 1
    if (i > 1) Omega[i, i - 1] <- rho
    if (i < p) Omega[i, i + 1] <- rho
}
Sigma <- solve(Omega)

# Signal Generation
S_0 <- 1:10
v <- rep(0, p)
v[S_0] <- 1
delta <- sqrt(9 / sum(v^2))
mu_diff <- as.numeric(Sigma %*% (v * delta))
mu1 <- mu_diff / 2
mu2 <- -mu_diff / 2
true_labels <- c(rep(1, n / 2), rep(2, n / 2))

acc_calc <- function(pred) {
    if (length(unique(pred)) < 2) {
        return(0.5)
    }
    # Handle potentially more than K clusters (e.g. from hdbscan)
    if (length(unique(pred)) > K) {
        # Tabulate to find best mapping for Accuracy
        tab <- table(pred, true_labels)
        return(sum(apply(tab, 1, max)) / n)
    }
    max(mean(pred == true_labels), mean(pred != true_labels))
}

results <- data.frame(
    rep = 1:n_reps,
    kernlab_acc = 0,
    hclust_ward_acc = 0,
    hclust_avg_acc = 0,
    hdbscan_acc = 0
)

cat(sprintf("Running %d repetitions (Diverse Methods Benchmark)...\n", n_reps))

for (i in 1:n_reps) {
    set.seed(42 + i)
    cat(sprintf("Rep %d... ", i))

    # Generate Data
    X1 <- mvrnorm(n / 2, mu1, Sigma)
    X2 <- mvrnorm(n / 2, mu2, Sigma)
    X <- t(rbind(X1, X2)) # p x n
    Data <- t(X) # n x p

    # 1. kernlab::specc (Current Best)
    results$kernlab_acc[i] <- acc_calc(tryCatch(as.integer(specc(Data, centers = K)), error = function(e) rep(1, n)))

    # 2. Hierarchical (Ward.D2)
    dist_mat <- dist(Data)
    hc_ward <- hclust(dist_mat, method = "ward.D2")
    results$hclust_ward_acc[i] <- acc_calc(cutree(hc_ward, k = K))

    # 3. Hierarchical (Average)
    hc_avg <- hclust(dist_mat, method = "average")
    results$hclust_avg_acc[i] <- acc_calc(cutree(hc_avg, k = K))

    # 4. HDBSCAN
    # minPts is usually low for n=500
    res_hdb <- hdbscan(Data, minPts = 10)
    # HDBSCAN can return 0 for outliers, we treat them as a cluster or map them
    clust_hdb <- res_hdb$cluster
    clust_hdb[clust_hdb == 0] <- max(clust_hdb) + 1 # Outliers as a separate group
    results$hdbscan_acc[i] <- acc_calc(clust_hdb)

    cat(sprintf(
        "KL: %.3f, Ward: %.3f, Avg: %.3f, HDB: %.3f\n",
        results$kernlab_acc[i], results$hclust_ward_acc[i],
        results$hclust_avg_acc[i], results$hdbscan_acc[i]
    ))
}

# Summary
cat("\n=== Diverse Methods Summary ===\n")
cat(sprintf("Mean kernlab Accuracy:  %.4f\n", mean(results$kernlab_acc)))
cat(sprintf("Mean hclust (Ward):     %.4f\n", mean(results$hclust_ward_acc)))
cat(sprintf("Mean hclust (Avg):      %.4f\n", mean(results$hclust_avg_acc)))
cat(sprintf("Mean HDBSCAN Accuracy:  %.4f\n", mean(results$hdbscan_acc)))

saveRDS(results, "diverse_methods_benchmark.rds")
