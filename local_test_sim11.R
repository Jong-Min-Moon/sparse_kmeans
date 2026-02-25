# local_test_sim11.R
source("code_r/sparse_symmetric_data_generator.R")
source("code_r/block_coordinate_optim_warmstart_tvs.R")
source("code_r/block_coordinate_optim_greedy_unknowncov_SAM.R")
source("code_r/selection_block_greedy_screening.R")
source("code_r/ESSC.R")
source("code_r/ISEE_residual_lasso.R")
source("code_r/get_intercept_residual_lasso.R")
source("code_r/get_cov_small.R")
source("code_r/ISEE_bicluster.R")
source("code_r/clustering_block_knowncov.R")
source("code_r/sdp_kmeans.R")
source("code_r/get_cluster_acc.R")
source("code_r/utils.R")

set.seed(123)
generator <- sparse_symmetric_data_generator(
    support = 1:5,
    separation = 3,
    dimension = 50,
    precision_sparsity = 2,
    conditional_correlation = 0.45,
    flip = FALSE
)
data_res <- generate_data_from_generator(generator, n = 50, seed = 123)

res <- block_coordinate_optim_warmstart_tvs(
    X = data_res$X,
    K = 2,
    n_iter_greedy = 2, # Just test short iterations
    n_iter_tvs = 2,
    n_perms_greedy = 10,
    fdr_target_greedy = 0.4,
    C = 0.5,
    p_val_threshold = 0.01,
    true_labels = data_res$labels
)

print(str(res))
cat("It works!\n")
