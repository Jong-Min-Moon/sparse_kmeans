# ------------------------------------------------------------------
# aggregate_results.R
# Fully dynamic aggregation of AR1 Conda simulation outputs.
# ------------------------------------------------------------------

# Ensure script runs in its own directory
args        <- commandArgs(trailingOnly = FALSE)
script_name <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_name) > 0 && file.exists(script_name)) {
    setwd(dirname(normalizePath(script_name)))
}

# ------------------------------------------------------------------
# Helper: detect method identifiers from column names
# ------------------------------------------------------------------
detect_methods <- function(df) {
    list(
        accuracy = sub("^accuracy_", "", grep("^accuracy_", names(df), value = TRUE)),
        runtime  = sub("^runtime_",  "", grep("^runtime_",  names(df), value = TRUE))
    )
}

# ------------------------------------------------------------------
# Helper: compute summary data.frame using base R only.
# ------------------------------------------------------------------
summarise_dynamic <- function(df, grouping_cols, methods) {

    # All unique combinations of grouping values, sorted
    key_df  <- unique(df[, grouping_cols, drop = FALSE])
    key_df  <- key_df[do.call(order, as.list(key_df)), , drop = FALSE]
    rownames(key_df) <- NULL

    extra_L_cols <- grep("_L$", names(df), value = TRUE)

    result_rows <- lapply(seq_len(nrow(key_df)), function(i) {
        # Build logical mask for this group
        mask <- rep(TRUE, nrow(df))
        for (col in grouping_cols) {
            mask <- mask & (df[[col]] == key_df[[col]][i])
        }
        sub_df <- df[mask, , drop = FALSE]

        row        <- as.list(key_df[i, , drop = FALSE])
        row$n_runs <- nrow(sub_df)

        for (m in methods$accuracy) {
            col <- paste0("accuracy_", m)
            if (col %in% names(sub_df)) {
                vals                               <- sub_df[[col]]
                row[[paste0("mean_acc_", m)]]      <- mean(vals, na.rm = TRUE)
                row[[paste0("sd_acc_",   m)]]      <- sd(vals,   na.rm = TRUE)
                row[[paste0("na_frac_",  m)]]      <- mean(is.na(vals))
            }
        }

        for (m in methods$runtime) {
            col <- paste0("runtime_", m)
            if (col %in% names(sub_df)) {
                row[[paste0("mean_rt_", m)]] <- mean(sub_df[[col]], na.rm = TRUE)
            }
        }

        for (col in extra_L_cols) {
            row[[paste0("mean_", col)]] <- mean(sub_df[[col]], na.rm = TRUE)
        }

        as.data.frame(row, stringsAsFactors = FALSE)
    })

    do.call(rbind, result_rows)
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

results_root <- "results_ar1"
if (!dir.exists(results_root)) {
    stop("'results_ar1/' directory not found. Run a simulation first.")
}

noise_types <- list.dirs(results_root, full.names = FALSE, recursive = FALSE)
noise_types <- noise_types[nchar(noise_types) > 0]

if (length(noise_types) == 0) {
    cat("No noise-type subdirectories found under results_ar1/. Nothing to aggregate.\n")
    quit(save = "no")
}

cat(sprintf("Discovered noise types: %s\n\n", paste(noise_types, collapse = ", ")))

for (noise in noise_types) {
    results_dir <- file.path(results_root, noise)

    files <- list.files(results_dir,
                        pattern    = "^sim_job\\d+_p\\d+\\.rds$",
                        full.names = TRUE)

    if (length(files) == 0) {
        cat(sprintf("Skipping '%s': no result files found.\n\n", noise))
        next
    }

    cat(sprintf("[%s] Loading %d files...\n", noise, length(files)))

    rows <- lapply(files, function(f) {
        tryCatch(readRDS(f), error = function(e) {
            warning(sprintf("Skipping corrupt file %s: %s", basename(f), e$message))
            NULL
        })
    })
    rows <- Filter(Negate(is.null), rows)

    if (length(rows) == 0) {
        cat(sprintf("Skipping '%s': all files failed to load.\n\n", noise))
        next
    }

    df <- do.call(rbind, rows)
    rownames(df) <- NULL

    methods       <- detect_methods(df)
    grouping_cols <- intersect(c("p", "sep"), names(df))
    p_found       <- sort(unique(df$p))
    extra_L_cols  <- grep("_L$", names(df), value = TRUE)

    cat(sprintf("[%s] Detected methods : %s\n",  noise, paste(methods$accuracy, collapse = ", ")))
    cat(sprintf("[%s] Detected p values: %s\n",  noise, paste(p_found,          collapse = ", ")))
    if (length(extra_L_cols) > 0) {
        cat(sprintf("[%s] Extra scalar cols: %s\n", noise, paste(extra_L_cols, collapse = ", ")))
    }

    rep_tbl <- table(df[, grouping_cols, drop = FALSE])
    cat(sprintf("[%s] Replicates per (p, sep):\n", noise))
    print(rep_tbl)

    summary_df <- summarise_dynamic(df, grouping_cols, methods)

    cat(sprintf("\n[%s] Accuracy / runtime summary:\n", noise))
    print(summary_df)

    # Note the _ar1 suffix to easily match the directory intent
    saveRDS(df,         file = sprintf("aggregated_ar1_%s.rds", noise))
    saveRDS(summary_df, file = sprintf("summary_ar1_%s.rds",    noise))
    write.csv(summary_df,
              file      = sprintf("summary_ar1_%s.csv", noise),
              row.names = FALSE)

    cat(sprintf("\n[%s] Saved: aggregated_ar1_%s.rds, summary_ar1_%s.{rds,csv}\n\n",
                noise, noise, noise))
}

cat("Aggregation complete.\n")
