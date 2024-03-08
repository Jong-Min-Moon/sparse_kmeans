function acc_vec = get_acc(cluster_est_mat, cluster_true)
    n_iter = size(cluster_est_mat, 1);
    acc_vec = zeros(n_iter, 1);
    for i = 1:n_iter
        cluster_est_now = cluster_est_mat(i,:);
        acc_vec(i) = max( mean(cluster_true == cluster_est_now), mean(cluster_true == -cluster_est_now));
    end
    