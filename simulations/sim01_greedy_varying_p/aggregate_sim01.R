# Aggregate Simulation 01 Results
# Loads all .rds files from all output_p* directories and creates a summary.

library(dplyr)
library(tidyr)
library(purrr)

# Set paths
base_dir <- "."
summary_file <- "summary_sim01.csv"

# Find all output_p directories
output_dirs <- list.dirs(base_dir, full.names = TRUE, recursive = FALSE)
output_dirs <- output_dirs[grepl("output_p", output_dirs)]

if (length(output_dirs) == 0) {
    stop("No output_p* directories found in ", base_dir)
}

cat(sprintf("Found %d output directories: %s\n", length(output_dirs), paste(basename(output_dirs), collapse = ", ")))

# List all RDS files across all directories
all_files <- unlist(lapply(output_dirs, function(d) {
    list.files(d, pattern = "\\.rds$", full.names = TRUE)
}))

if (length(all_files) == 0) {
    stop("No .rds files found in any output_p* directory.")
}

cat(sprintf("Found %d result files in total. Aggregating...\n", length(all_files)))

# Load and combine data
results_list <- map(all_files, function(f) {
    res_data <- tryCatch(
        {
            readRDS(f)
        },
        error = function(e) {
            warning(sprintf("Error reading %s: %s", f, e$message))
            NULL
        }
    )

    if (is.null(res_data)) {
        return(NULL)
    }

    # Flatten structure
    data.frame(
        p = res_data$p,
        rep = res_data$rep,
        time = res_data$metrics$time,
        ari = res_data$metrics$ari,
        acc = res_data$metrics$acc,
        iterations = res_data$metrics$iterations
    )
})

all_results <- bind_rows(results_list)

# Calculate summaries by p
summary_stats <- all_results %>%
    group_by(p) %>%
    summarise(
        n_reps = n(),
        mean_ari = mean(ari, na.rm = TRUE),
        sd_ari = sd(ari, na.rm = TRUE),
        mean_acc = mean(acc, na.rm = TRUE),
        sd_acc = sd(acc, na.rm = TRUE),
        mean_time = mean(time, na.rm = TRUE),
        mean_iter = mean(iterations, na.rm = TRUE)
    ) %>%
    arrange(p)

# Save summary
write.csv(summary_stats, summary_file, row.names = FALSE)
cat(sprintf("Summary saved to %s\n", summary_file))

# Print summary to console
print(summary_stats)
