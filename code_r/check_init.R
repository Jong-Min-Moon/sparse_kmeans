library(stats)
library(mclust)

# Source ADMM solver
if (file.exists("code_r/sdp_kmeans.R")) {
    source("code_r/sdp_kmeans.R")
} else {
    source("../../code_r/sdp_kmeans.R")
}

# Parameters matching poor performance case
n <- 200
K <- 2
s <- 10
p <- 3000
mu_val <- sqrt(4 / s)

cat("Investigating Initialization Quality\n")

# Generate Data (Consistent Seed)
set.seed(4001)
n1 <- n / 2
n2 <- n / 2
mu1 <- numeric(p)
mu2 <- numeric(p)
mu1[1:s] <- mu_val
mu2[1:s] <- -mu_val

X1 <- matrix(rnorm(n1 * p), nrow = p) + mu1
X2 <- matrix(rnorm(n2 * p), nrow = p) + mu2
X <- cbind(X1, X2)
G <- crossprod(X)
true_labels <- c(rep(1, n1), rep(2, n2))

cat("Running Spectral Initialization...\n")
init_decomp <- RSpectra::eigs_sym(G, K, which = "LA")
V_init <- init_decomp$vectors
km_init <- kmeans(V_init, centers = K, nstart = 20)

acc_init <- max(mean(km_init$cluster == true_labels), mean(km_init$cluster != true_labels))
cat(sprintf("Initialization Accuracy: %.2f\n", acc_init))
