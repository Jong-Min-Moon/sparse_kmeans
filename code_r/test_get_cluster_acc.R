source("D:/GitHub/sparse_kmeans/code_r/get_cluster_acc.R")

cat("Testing get_cluster_acc...\n")

# Case 1: Perfect match (Acc = 1.0)
true_1 <- c(1, 1, 2, 2, 3, 3)
est_1 <- c(1, 1, 2, 2, 3, 3)
acc_1 <- get_cluster_acc(est_1, true_1)
cat(sprintf("Case 1 (Perfect): Expected 1.0, Got %.4f\n", acc_1))

# Case 2: Permuted (Acc = 1.0)
est_2 <- c(2, 2, 3, 3, 1, 1)
acc_2 <- get_cluster_acc(est_2, true_1)
cat(sprintf("Case 2 (Permuted): Expected 1.0, Got %.4f\n", acc_2))

# Case 3: Partial mismatch (Acc should be 5/6 = 0.8333)
est_3 <- c(1, 2, 2, 2, 3, 3) # One mistake in cluster 1 (index 2 is 2 instead of 1)
acc_3 <- get_cluster_acc(est_3, true_1)
cat(sprintf("Case 3 (Mismatch): Expected 0.8333, Got %.4f\n", acc_3))

# Case 4: Different number of clusters (3 true vs 2 estimated)
est_4 <- c(1, 1, 1, 1, 2, 2)
acc_4 <- get_cluster_acc(est_4, true_1)
cat(sprintf("Case 4 (K mismatch): Got %.4f\n", acc_4))

if (abs(acc_1 - 1.0) < 1e-6 && abs(acc_2 - 1.0) < 1e-6 && abs(acc_3 - 5 / 6) < 1e-6) {
    cat("\nSUCCESS: All basic tests passed.\n")
} else {
    cat("\nFAILURE: Some tests failed.\n")
}
