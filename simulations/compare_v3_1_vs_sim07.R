v3_1_dir <- "simulations/sim_v3_1_oracle_isee/output"
sim07_dir <- "simulations/sim07_permutation_fdr_0.4/output"

cat("=== Clustering Accuracy Comparison ===\n\n")
cat(sprintf("%-8s  %-20s  %-20s\n", "Job ID", "v3.1 Oracle ISEE", "Sim07 Perm FDR 0.4"))
cat(paste(rep("-", 52), collapse = ""), "\n")

v3_1_accs <- numeric(10)
sim07_accs <- numeric(10)
v3_1_aris <- numeric(10)
sim07_aris <- numeric(10)
v3_1_tp <- numeric(10)
sim07_tp <- numeric(10)
v3_1_fp <- numeric(10)
sim07_fp <- numeric(10)

for (i in 1:10) {
    r1 <- readRDS(file.path(v3_1_dir, sprintf("sim_id%d.rds", i)))
    r2 <- readRDS(file.path(sim07_dir, sprintf("sim_id%d.rds", i)))
    v3_1_accs[i] <- r1[["acc"]]
    sim07_accs[i] <- r2[["acc"]]
    v3_1_aris[i] <- r1[["ari"]]
    sim07_aris[i] <- r2[["ari"]]
    v3_1_tp[i] <- r1[["tp"]]
    sim07_tp[i] <- r2[["tp"]]
    v3_1_fp[i] <- r1[["fp"]]
    sim07_fp[i] <- r2[["fp"]]
    cat(sprintf("%-8d  %-20.4f  %-20.4f\n", i, v3_1_accs[i], sim07_accs[i]))
}
cat(paste(rep("-", 52), collapse = ""), "\n")
cat(sprintf("%-8s  %-20.4f  %-20.4f\n", "Mean", mean(v3_1_accs), mean(sim07_accs)))
cat(sprintf("%-8s  %-20.4f  %-20.4f\n", "SD", sd(v3_1_accs), sd(sim07_accs)))

cat("\n=== ARI Comparison ===\n\n")
cat(sprintf("%-8s  %-20s  %-20s\n", "Job ID", "v3.1 Oracle ISEE", "Sim07 Perm FDR 0.4"))
cat(paste(rep("-", 52), collapse = ""), "\n")
for (i in 1:10) {
    cat(sprintf("%-8d  %-20.4f  %-20.4f\n", i, v3_1_aris[i], sim07_aris[i]))
}
cat(paste(rep("-", 52), collapse = ""), "\n")
cat(sprintf("%-8s  %-20.4f  %-20.4f\n", "Mean", mean(v3_1_aris), mean(sim07_aris)))
cat(sprintf("%-8s  %-20.4f  %-20.4f\n", "SD", sd(v3_1_aris), sd(sim07_aris)))

cat("\n=== Feature Selection (TP / FP) ===\n\n")
cat(sprintf("%-8s  %-12s  %-12s  %-12s  %-12s\n", "Job ID", "v3.1 TP", "v3.1 FP", "s07 TP", "s07 FP"))
cat(paste(rep("-", 60), collapse = ""), "\n")
for (i in 1:10) {
    cat(sprintf("%-8d  %-12d  %-12d  %-12d  %-12d\n", i, v3_1_tp[i], v3_1_fp[i], sim07_tp[i], sim07_fp[i]))
}
cat(paste(rep("-", 60), collapse = ""), "\n")
cat(sprintf("%-8s  %-12.1f  %-12.1f  %-12.1f  %-12.1f\n", "Mean", mean(v3_1_tp), mean(v3_1_fp), mean(sim07_tp), mean(sim07_fp)))
