# Profile glmnet families
library(glmnet)

n <- 100
p <- 400
K <- 2
y1 <- rnorm(n)
y2 <- rnorm(n)
Y <- cbind(y1, y2)
X <- matrix(rnorm(n * p), n, p)
Z <- matrix(0, n, K)
Z[1:(n/2), 1] <- 1
Z[(n/2+1):n, 2] <- 1
D <- cbind(Z, X)
p_fac <- c(rep(0, K), rep(1, p))

cat("\n--- Profiling glmnet calls ---\n")

# 1. Standard Gaussian (1 call, N samples)
t1 <- Sys.time()
fit1 <- cv.glmnet(X, y1)
t2 <- Sys.time()
cat(sprintf("Single Gaussian (N=%d, P=%d): %.4f s\n", n, p, as.numeric(difftime(t2, t1, units = "secs"))))

# 2. Mgaussian (1 call, N samples)
t1 <- Sys.time()
fit2 <- cv.glmnet(D, Y, family = "mgaussian", penalty.factor = p_fac, intercept=FALSE)
t2 <- Sys.time()
cat(sprintf("Single Mgaussian (N=%d, P=%d): %.4f s\n", n, p, as.numeric(difftime(t2, t1, units = "secs"))))

# 3. Independent Gaussian (2 calls, N/2 samples)
# This mimics the original ISEE loop over clusters
t1 <- Sys.time()
fit3a <- cv.glmnet(X[1:(n/2), ], y1[1:(n/2)])
fit3b <- cv.glmnet(X[(n/2+1):n, ], y1[(n/2+1):n])
fit3c <- cv.glmnet(X[1:(n/2), ], y2[1:(n/2)])
fit3d <- cv.glmnet(X[(n/2+1):n, ], y2[(n/2+1):n])
t2 <- Sys.time()
cat(sprintf("4x Gaussian (N=%d, P=%d): %.4f s\n", n/K, p, as.numeric(difftime(t2, t1, units = "secs"))))

cat("\nConclusion: Compare mgaussian vs 4x Gaussian.\n")
