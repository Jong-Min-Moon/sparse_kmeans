  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(latex2exp)
  
  # 1. Load and clean
  data <- read.csv("/Users/jmmoon/Documents/GitHub/sparse_kmeans/experiment/34_01_16_2025/plot/trajectory_results.csv")
  # Manual naming to ensure consistency with the rest of the script
  colnames(data) <- c("iter", "model", "mean_acc", "mean_tp", "mean_fp", "std_acc") 
  
  # 2. Reshape and Update Labels
  data_long <- data %>%
    filter(as.integer(iter) %% 5 == 0) %>%
    mutate(model = as.factor(model)) %>%
    pivot_longer(
      cols = c(mean_acc, mean_tp, mean_fp),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(metric = factor(case_when(
      metric == "mean_acc" ~ "Average Accuracy",
      metric == "mean_tp"  ~ "Average True Positives",
      metric == "mean_fp"  ~ "Average False Positives"
    ), levels = c("Average Accuracy", "Average True Positives", "Average False Positives")))
  
  # 3. Plotting
  legend_title <- TeX("Hyperparameter $C$")
  
  p <- ggplot(data_long, aes(x = iter, y = value, group = model, color = model)) +
    # Force solid lines for all models
    geom_line(linewidth = 0.6, linetype = "solid") + 
    facet_wrap(~metric, ncol = 3, scales = "free_y") + 
    # Using an academic-safe high-contrast palette (Set1 or Viridis)
    scale_color_brewer(palette = "Set2") + 
    labs(
      x = TeX("Iteration ($t$)"),
      y = "Average Value",
      color = legend_title
    ) +
    theme_bw() + 
    theme(
      text = element_text(family = "serif", size = 10),
      strip.background = element_blank(), 
      strip.text = element_text(face = "italic", size = 10),
      panel.grid.major = element_blank(), 
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.key.width = unit(1.0, "cm"),
      axis.ticks.length = unit(-0.12, "cm"), # Classic inward ticks
      axis.text.x = element_text(margin = margin(t = 8)),
      axis.text.y = element_text(margin = margin(r = 8)),
      panel.border = element_rect(colour = "black", fill=NA, linewidth=0.5)
    )
  
  # 4. Save in Annals format
  ggsave(
    #filename = "/Users/jmmoon/Documents/GitHub/sparse_kmeans/experiment/34_01_16_2025/plot/trajectory_annals_final.pdf", 
    filename = "/Users/jmmoon/Documents/GitHub/sparse_kmeans/experiment/34_01_16_2025/plot/trajectory_C.pdf", 
    plot = p,
    device = cairo_pdf, 
    width = 6.5,     
    height = 2.8,    
    units = "in",
    dpi = 600
  )
