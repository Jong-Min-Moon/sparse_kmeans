function run_bandit_task_save_mat(task_id, param_file, results_dir)
    % Load parameters
    params = readmatrix(param_file);
    this_param = params(task_id, :);
    C = this_param(1);
    p = this_param(2);
    rep = this_param(3);

    % --- 1. Generate or load dataset ---
    n = 100; % number of samples (adjust as needed)
    X = randn(p, n); % placeholder data
    K = 2;           % number of clusters
    cluster_true = randi([1 K], 1, n); % placeholder true labels
    support = true(1,p); % placeholder true signal, replace with real support if known

    % --- 2. Initialize and run bandit ---
    bandit = sdp_kmeans_bandit(X, K, C);
    n_iter = 200;

    % Preallocate trajectories
    alpha_trajectory = zeros(n_iter, p);
    beta_trajectory  = zeros(n_iter, p);
    tp_trajectory    = zeros(n_iter,1);
    fp_trajectory    = zeros(n_iter,1);
    acc_trajectory   = zeros(n_iter,1);

    fprintf('Running simulation for C=%.1f, p=%d, rep=%d\n', C, p, rep);

    % --- 3. Simulation Loop ---
    bandit.set_bayesian_parameters();
    bandit.initialize_cluster_est();
    bandit.entries_survived = zeros(n_iter, p);

    for i = 1:n_iter
        % 1. Thompson Sampling selection
        variable_subset_now = bandit.choose();
        bandit.entries_survived(i,:) = variable_subset_now;

        % 2. Reward
        reward_now = bandit.reward(variable_subset_now, i);

        % 3. Update Beta parameters
        bandit.update(variable_subset_now, reward_now);

        % 4. Record trajectories
        alpha_trajectory(i,:) = bandit.alpha;
        beta_trajectory(i,:)  = bandit.beta;

        % 5. TP/FP per iteration
        [tp, fp, ~, ~] = bandit.evaluate_discovery(support);
        tp_trajectory(i) = tp(i);
        fp_trajectory(i) = fp(i);

        % 6. Accuracy per iteration
        if isKey(bandit.acc_dict, i)
            acc_trajectory(i) = bandit.acc_dict(i);
        else
            acc_trajectory(i) = NaN;
        end
    end

    % --- 4. Final clustering ---
    final_selection = bandit.signal_entry_est;
    X_sub_final = bandit.X(final_selection,:);
    bandit.cluster_est = bandit.get_cluster(X_sub_final, K);

    % Record final accuracy
    bandit.evaluate_accuracy(bandit.cluster_est, cluster_true, n_iter+1);
    acc_trajectory = [acc_trajectory; bandit.acc_dict(n_iter+1)];

    % --- 5. Save results ---
    fname = fullfile(results_dir, sprintf('res_C%0.1f_p%d_rep%d.mat', C, p, rep));
    save(fname, 'alpha_trajectory', 'beta_trajectory', 'tp_trajectory', ...
               'fp_trajectory', 'acc_trajectory');

    fprintf('Saved %s\n', fname);
end
