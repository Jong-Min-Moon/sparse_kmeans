# Non-Isotropic Visualization
source("D:/GitHub/sparse_kmeans/code_r/data_generator.R")

# Parameters: p=3, n=500, rho=0.8 (Stronger for visual effect), sep=6
p <- 3
n <- 500
rho <- 0.45
sep <- 8

cat("Generating 3D data with strong correlation...\n")
generator <- get_specification_chaingraph(
    support = 1:3, # Features 1 and 2 are signal
    separation = sep,
    dimension = p,
    precision_sparsity = 2, # Higher sparsity to see more correlation links
    conditional_correlation = rho,
    flip = FALSE
)

data_res <- generate_data_from_specification(generator, n, seed = 42)
X <- t(data_res$X) # n x p
labels <- data_res$labels

# Plotting: Use Pairs plot for non-isotropy check
png("data_nonisotropic_check.png", width = 1000, height = 1000)
colors <- c(adjustcolor("#1f77b4", alpha.f = 0.6), adjustcolor("#ff7f0e", alpha.f = 0.6))

# Pairs plot shows all correlations
pairs(X,
    col = colors[labels], pch = 19,
    main = sprintf("Non-Isotropic Gaussian Mixture (rho=%.2f, sep=%.1f)", rho, sep)
)
dev.off()

# Also a specific 2D tilt plot (X1 vs X2)
png("data_2d_tilt.png", width = 600, height = 600)
plot(X[, 1], X[, 2],
    col = colors[labels], pch = 19,
    main = "Tilt Visualization (X1 vs X2)",
    xlab = "Feature 1", ylab = "Feature 2"
)
# Add a line to show the covariance direction if clear
dev.off()

cat("Plots saved to data_nonisotropic_check.png and data_2d_tilt.png\n")
# Print the covariance matrix
cat("\nTheoretical Covariance Matrix (Subset):\n")
print(round(generator$covariance_matrix, 3))
cat("\nSample Covariance Matrix:\n")
print(round(cov(X), 3))
