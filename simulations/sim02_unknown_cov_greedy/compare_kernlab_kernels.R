# Benchmark: kernlab specc Kernels
library(MASS)
library(mclust)
library(kernlab)

# Simulation Parameters
p <- 400
n <- 500
K <- 2
rho <- 0.45
n_reps <- 30

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

kernels <- c("rbfdot", "polydot", "vanilladot", "tanhdot", "laplacedot", "anovadot")
results <- as.data.frame(matrix(0, n_reps, length(kernels)))
colnames(results) <- kernels
results$rep <- 1:n_reps

cat(sprintf("Running %d repetitions for %d kernels...\n", n_reps, length(kernels)))

for (i in 1:n_reps) {
    set.seed(42 + i)
    cat(sprintf("Rep %d... ", i))

    # Generate Data
    X1 <- mvrnorm(n / 2, mu1, Sigma)
    X2 <- mvrnorm(n / 2, mu2, Sigma)
    Data <- rbind(X1, X2) # n x p

    for (k_name in kernels) {
        res <- tryCatch(
            {
                as.integer(specc(Data, centers = K, kernel = k_name))
            },
            error = function(e) rep(1, n)
        )
        results[i, k_name] <- acc_calc(res)
    }

    # Print progress for a few
    cat(sprintf(
        "RBF: %.3f, Poly: %.3f, Linear: %.3f\n",
        results[i, "rbfdot"], results[i, "polydot"], results[i, "vanilladot"]
    ))
}

# Summary
cat("\n=== kernlab Kernel Comparison Summary ===\n")
summary_stats <- data.frame(
    Kernel = kernels,
    Mean_Acc = colMeans(results[, kernels]),
    SD = apply(results[, kernels], 2, sd)
)
print(summary_stats)

saveRDS(results, "kernlab_kernel_comparison.rds")
