# Benchmark: Tuning sigma for Laplacian Kernel in specc
library(MASS)
library(mclust)
library(kernlab)

# Simulation Parameters
p <- 400
n <- 500
K <- 2
rho <- 0.45
n_reps <- 20

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

# Range of sigma values to test
sigmas <- c(0.001, 0.01, 0.1, 1, 10, 100)
results <- as.data.frame(matrix(0, n_reps, length(sigmas)))
colnames(results) <- paste0("sigma_", sigmas)
results$rep <- 1:n_reps

cat(sprintf("Running %d repetitions for %d sigma values...\n", n_reps, length(sigmas)))

for (i in 1:n_reps) {
    set.seed(42 + i)
    cat(sprintf("Rep %d... ", i))

    # Generate Data
    X1 <- mvrnorm(n / 2, mu1, Sigma)
    X2 <- mvrnorm(n / 2, mu2, Sigma)
    Data <- rbind(X1, X2) # n x p

    for (s_val in sigmas) {
        col_name <- paste0("sigma_", s_val)
        res <- tryCatch(
            {
                # Explicitly create kernel object
                k_func <- laplacedot(sigma = s_val)
                as.integer(specc(Data, centers = K, kernel = k_func))
            },
            error = function(e) rep(1, n)
        )
        results[i, col_name] <- acc_calc(res)
    }
    cat("done\n")
}

# Summary
cat("\n=== Laplacian sigma Tuning Summary ===\n")
summary_stats <- data.frame(
    Sigma = sigmas,
    Mean_Acc = colMeans(results[, 1:length(sigmas)]),
    SD = apply(results[, 1:length(sigmas)], 2, sd)
)
print(summary_stats)

saveRDS(results, "laplacian_sigma_tuning.rds")
