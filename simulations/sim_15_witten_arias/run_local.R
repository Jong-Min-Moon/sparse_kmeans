# Local parallel execution script for sim_15_witten_arias
library(foreach)
library(doParallel)

# Ensure we are in the script's directory
args <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0 && file.exists(script_name)) {
    setwd(dirname(normalizePath(script_name)))
}

# Number of iterations to run
n_runs <- 100
# Target separations
separations <- c(4, 5)

# Setup parallel backend to use multiple cores
# Adjust the number of cores based on your CPU
n_cores <- parallel::detectCores() - 1
if (n_cores < 1) n_cores <- 1

cl <- makeCluster(n_cores)
registerDoParallel(cl)

cat(sprintf("Starting local execution using %d cores...\n", n_cores))

# Create directories if they don't exist
dir.create("logs", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# Run simulations in parallel
start_time <- Sys.time()

for (sep in separations) {
    cat(sprintf("\n--- Starting jobs for separation = %d ---\n", sep))

    # Run the driver in parallel for 1 to n_runs
    res <- foreach(job_id = 1:n_runs, .combine = c) %dopar% {
        # Construct the command to run the driver
        cmd <- sprintf("Rscript driver.R --job_id %d --sep %d", job_id, sep)

        # Run the command
        out <- system(cmd, intern = TRUE, ignore.stderr = FALSE)

        # Optionally write log to file if needed (append mode)
        # writeLines(out, sprintf("logs/sim_id%d_sep%d.log", job_id, sep))

        return(0)
    }
}

stopCluster(cl)

end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units = "mins"))

cat(sprintf("\nLocal execution completed in %.2f minutes.\n", runtime))
cat("You can now run 'Rscript aggregate_sim15.R' to aggregate the results.\n")
