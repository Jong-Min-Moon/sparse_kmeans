library(ggplot2)

# Load data
all_results <- readRDS("all_results_sim02.rds")

# Create plot
p <- ggplot(all_results, aes(x = accuracy)) +
    geom_histogram(bins = 20, fill = "#4A90E2", color = "white", alpha = 0.8) +
    geom_vline(aes(xintercept = mean(accuracy)), color = "#D0021B", linetype = "dashed", size = 1) +
    theme_minimal(base_size = 14) +
    labs(
        title = "Distribution of Clustering Accuracy",
        subtitle = "Simulation 02: Unknown Covariance (Greedy)",
        x = "Accuracy",
        y = "Frequency",
        caption = sprintf("Mean Accuracy: %.4f (N=100)", mean(all_results$accuracy))
    ) +
    theme(
        plot.title = element_text(face = "bold", size = 18),
        plot.subtitle = element_text(color = "grey40"),
        panel.grid.minor = element_blank(),
        axis.title = element_text(face = "bold")
    )

# Save plot
ggsave("accuracy_histogram.png", p, width = 8, height = 6, dpi = 300)
cat("Plot saved as accuracy_histogram.png\n")
