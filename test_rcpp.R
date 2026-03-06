library(Rcpp)
library(RcppArmadillo)

# Source the R implementation
source("code_r/hardt_price_gmm_1d.R")

# Compile and source the C++ implementation
cat("Compiling C++ implementation...\n")
sourceCpp("code_rcpp/hardt_price_1d.cpp")

cat("Success! Testing recovery on synthetic data...\n")

set.seed(42)
n <- 10000
x <- c(rnorm(n * 0.3, mean = -2, sd = 0.5), rnorm(n * 0.7, mean = 2, sd = 1.0))

cat("\n--- Running R Implementation ---\n")
res_r <- Recover1DMixture(x, delta = 0.05)
print(res_r)

cat("\n--- Running C++ Implementation ---\n")
res_cpp <- Recover1DMixture_cpp(x, delta = 0.05)
print(res_cpp)

cat("\n--- Differences ---\n")
cat("Comp1 mu diff:", abs(res_r$comp1$mu - res_cpp$comp1$mu), "\n")
cat("Comp2 mu diff:", abs(res_r$comp2$mu - res_cpp$comp2$mu), "\n")
