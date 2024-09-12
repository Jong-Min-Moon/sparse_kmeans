function [Omega_best, idx, score_vec] = glasso_bicluster(dg, n_lambda)
    x_noisy = dg.data;
    p = dg.dimension;
    n = dg.sample_size;
    cluster_est_now = dg.cluster_info_vec;

    X_g1_now = x_noisy(:, (cluster_est_now ==  1)); 
    X_g2_now = x_noisy(:, (cluster_est_now ==  2));

    mean_g1_now = mean(X_g1_now, 2);
    mean_g2_now = mean(X_g2_now, 2);
    data_centered_np = [(X_g1_now - mean_g1_now) (X_g2_now - mean_g2_now)]';
    sd_vec  = std(data_centered_np);

    sample_stan = bsxfun(@rdivide, data_centered_np , sd_vec  );
    cv_partition = cvpartition(n,"Kfold", 5);
    lambda_grid = (1:n_lambda)/n_lambda;
    score_vec = repelem(0, length(lambda_grid));
    for i = 1:5
        data_train = sample_stan(training(cv_partition, i), :);
        data_test = sample_stan(test(cv_partition, i), :);
        [wList, thetaList, lambdaList, errors] = GraphicalLassoPath(data_train , lambda_grid);
        for j = 1:length(lambda_grid)
            Omega_est = thetaList(:,:,j);
            loglik_out_sample = loglik_glasso(Omega_est, data_test);
            score_vec(j) = score_vec(j) + loglik_out_sample;
        end
    end
    [max_elem,idx] = max(score_vec);
    Omega_best = bsxfun(@rdivide, thetaList(:,:,idx), sd_vec);
    

