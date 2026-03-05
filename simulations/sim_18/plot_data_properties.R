# Source Generator
source("../../code_r/sparse_symmetric_data_generator.R")
library(ggplot2)
library(dplyr)

# ---------------------------------------------------------
# Data Generation Parameters
# ---------------------------------------------------------
p <- 400
n <- 200 
K <- 2
rho <- 0.45
precision_sparsity <- 2
support <- 1:10
flip <- FALSE
separation <- 6

set.seed(2025)

generator <- sparse_symmetric_data_generator(
    support = support,
    separation = separation,
    dimension = p,
    precision_sparsity = precision_sparsity,
    conditional_correlation = rho,
    flip = flip
)

data_res <- generate_data_from_generator(generator, n, seed = 2025)
X <- data_res$X
true_labels <- data_res$labels

# Compute coordinate-wise properties
properties <- data.frame(
  Feature = 1:p,
  Variance = apply(X, 2, var),
  MeanDiff = abs(colMeans(X[true_labels == 1, ]) - colMeans(X[true_labels == 2, ]))
)

# Identify support features (1 to 10)
properties$IsSupport <- ifelse(properties$Feature %in% support, "Support", "Noise")

# Prepare long format for plotting
properties_long <- data.frame(
  Feature = rep(properties$Feature, 2),
  IsSupport = rep(properties$IsSupport, 2),
  Metric = rep(c("1. Coordinate-wise Variance", "2. Magnitude of Mean Difference"), each = nrow(properties)),
  Value = c(properties$Variance, properties$MeanDiff)
)

# Plot
p_plot <- ggplot(properties_long, aes(x = Feature, y = Value)) +
  geom_col(fill = "steelblue") +
  facet_wrap(~ Metric, scales = "free_y", ncol = 1) +
  theme_bw() +
  labs(
    title = "Sim 18: Data Generation Properties by Coordinate",
    x = "Coordinate Number (Feature Index)",
    y = "Value"
  )

ggsave("coordinate_properties.png", p_plot, width = 10, height = 7, dpi = 300)

cat("Successfully generated plot: simulations/sim_18/coordinate_properties.png\n")
