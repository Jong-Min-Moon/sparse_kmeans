# Comparison: Spectral Clustering vs PAM
library(MASS)
library(mclust)
library(kernlab)
library(cluster)

# Simulation Parameters
p <- 400
n <- 500
K <- 2
rho <- 0.45
n_reps <- 50

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
    max(mean(pred == true_labels), mean(pred != true_labels))
}

results <- data.frame(
    rep = 1:n_reps,
    spectral_acc = 0,
    pam_acc = 0
)

cat(sprintf("Running %d repetitions (Spectral vs PAM)...\n", n_reps))

for (i in 1:n_reps) {
    set.seed(42 + i)

    # Generate Data
    X1 <- mvrnorm(n / 2, mu1, Sigma)
    X2 <- mvrnorm(n / 2, mu2, Sigma)
    X <- t(rbind(X1, X2)) # p x n
    Data <- t(X) # n x p

    # 1. Spectral Clustering (kernlab::specc)
    results$spectral_acc[i] <- acc_calc(tryCatch(as.integer(specc(Data, centers = K)), error = function(e) rep(1, n)))

    # 2. PAM (cluster::pam)
    results$pam_acc[i] <- acc_calc(as.integer(pam(Data, k = K)$clustering))

    if (i %% 5 == 0) cat(sprintf("Completed %d/%d reps...\n", i, n_reps))
}

# Summary
cat("\n=== Comparison Summary (50 reps) ===\n")
cat(sprintf("Mean Spectral Accuracy: %.4f (sd: %.4f)\n", mean(results$spectral_acc), sd(results$spectral_acc)))
cat(sprintf("Mean PAM Accuracy:      %.4f (sd: %.4f)\n", mean(results$pam_acc), sd(results$pam_acc)))

# T-test
ttest <- t.test(results$spectral_acc, results$pam_acc, paired = TRUE)
cat(sprintf("P-value (paired t-test): %.4f\n", ttest$p.value))

saveRDS(results, "spec_pam_comparison_results.rds")
