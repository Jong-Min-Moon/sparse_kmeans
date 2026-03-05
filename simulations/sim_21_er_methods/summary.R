# load the aggregated results RDS
results <- readRDS("results_aggregated_sepNULL.rds")
# Results has Job_ID, p, n, sep, s, ... and accuracy/features for methods
n_runs <- nrow(results)

mean_acc_witten <- mean(results$accuracy_witten)
mean_feat_witten <- mean(results$accuracy_witten) # Witten accuracy vs feature is stored in L but we can read it off intermediate logs easily
# Actually it's better to just read accuracy since only accuracy/runtime is stored in dataframe, IFPCA stores L

cat("\n------------ SIMULATION 21 RESULTS ------------\n")
cat("Erdos Renyi Data: n=200, p=200, sep=Auto(NULL), s=10\n\n")

cat(sprintf("Witten     Avg Accuracy: %.3f\n", mean(results$accuracy_witten)))
cat(sprintf("Arias      Avg Accuracy: %.3f\n", mean(results$accuracy_arias)))
cat(sprintf("IF-PCA     Avg Accuracy: %.3f\n", mean(results$accuracy_ifpca)))
cat("-----------------------------------------------\n")
