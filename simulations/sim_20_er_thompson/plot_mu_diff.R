# Plot magnitude of mu_1 - mu_2 for Erdos Renyi Data
# Setting based on simulations/sim_20_er_thompson/simulation.R

source("../../code_r/sparse_symmetric_data_generator.R")

# Data Generation Parameters from simulation.R
p <- 200
n <- 200
sep_target <- NULL 
s <- 10

set.seed(1) # Using rep_id 1

generator_res <- generate_erdos_renyi_data(
    n = n,
    p = p,
    separation = sep_target,
    s = s
)

mu1 <- generator_res$mu1
mu2 <- generator_res$mu2

# Magnitude of difference
diff_magnitude <- abs(mu1 - mu2)

# Create the plot
png("mu_diff_magnitude.png", width = 800, height = 500, res = 100)
plot(1:p, diff_magnitude, 
     type = "h", 
     lwd = 2,
     col = "blue",
     xlab = "Feature Index (1 to p)", 
     ylab = "Magnitude |mu_1 - mu_2|", 
     main = "Magnitude of Mean Difference between Clusters",
     pch = 16)
points(1:p, diff_magnitude, col = "black", pch = 16, cex = 0.6)

# Highlight true support (1:s)
abline(v = s + 0.5, col = "red", lty = 2)
legend("topright", legend = c("True Support (1:10)", "Coordinates"), 
       col = c("red", "blue"), lty = c(2, 1), lwd = c(1, 2), pch = c(NA, 16))

dev.off()

cat("Saved plot to mu_diff_magnitude.png\n")
