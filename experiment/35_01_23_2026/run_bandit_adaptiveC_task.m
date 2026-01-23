function run_bandit_adaptiveC_task(rep, C_min, C_max, p_val, n_iter, results_dir)
    addpath(genpath('/home1/jongminm/sparse_kmeans'));

    n = 200;
    K = 2;
    s = 10;
    support = 1:s;

    rng(rep);

    [data, label_true] = generate_gaussian_data(n, p_val, s, 4, ...
        'iso', 'equal_symmetric', 0, rep, 0.5, rep);

    bandit = sdp_kmeans_bandit_adaptiveC(data', K, C_min, C_max);
    bandit.fit_predict(n_iter, label_true');

    beta_file = fullfile(results_dir, sprintf('bandit_adaptiveC_beta_rep%d.mat', rep));
    bandit.save_beta_params(beta_file);

    bandit.save_results(results_dir, rep, 4, support);

    fprintf('Replicate %d completed.\n', rep);
end
