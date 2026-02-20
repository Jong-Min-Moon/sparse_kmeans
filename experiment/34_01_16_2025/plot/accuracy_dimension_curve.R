library(ggplot2)
library(dplyr)
library(latex2exp)

# 1. Load the data
data_text <- "acc|dim|model
0.851571878121878|6000.0|0.2
0.81621013986014|6000.0|0.4
0.580635764235764|6000.0|0.6
0.571536463536464|6000.0|0.8
0.535225324675325|8000.0|0.01
0.729241258741259|8000.0|0.1
0.803811488511489|8000.0|0.2
0.764874825174825|8000.0|0.4
0.560973026973027|8000.0|0.6
0.556700549450549|8000.0|0.8
0.753302097902098|10000.0|0.2
0.735035814185814|10000.0|0.4
0.5492501998002|10000.0|0.6
0.555734865134865|10000.0|0.8"

df <- read.table(text = data_text, sep = "|", header = TRUE)

# 2. Prepare data
# Ensure model is a factor for categorical coloring
df_plot <- df %>%
  mutate(model = as.factor(model))

# 3. Plotting
legend_title <- TeX("Hyperparameter $C$")

p <- ggplot(df_plot, aes(x = dim, y = acc, group = model, color = model)) +
  # Lines and points (points help identify the specific dimensions tested)
  geom_line(linewidth = 0.6, linetype = "solid") + 
  geom_point(size = 1.5) +
  # Using the academic-safe Set2 palette as requested
  scale_color_brewer(palette = "Set2") + 
  labs(
    x = TeX("Dimension ($p$)"),
    y = "Average Accuracy",
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
    axis.ticks.length = unit(-0.12, "cm"), 
    axis.text.x = element_text(margin = margin(t = 8)),
    axis.text.y = element_text(margin = margin(r = 8)),
    panel.border = element_rect(colour = "black", fill=NA, linewidth=0.5)
  )

# 4. Save the plot
ggsave(
  filename = "/Users/jmmoon/Documents/GitHub/sparse_kmeans/experiment/34_01_16_2025/dimension_accuracy_curve.pdf", 
  plot = p,
  device = cairo_pdf, 
  width = 4.5,      # Adjusted width for a single-panel plot
  height = 3.5,    
  units = "in",
  dpi = 600
)