classdef sdp_kmeans_bandit_adaptiveC < handle
    % SDP_KMEANS_BANDIT_ADAPTIVEC Thompson Sampling Sparse K-Means with adaptive C

    properties
        X                   % Data matrix (d x n)
        K                   % Number of clusters
        n                   % Number of data points
        p                   % Data dimension
        alpha               % Alpha parameters of Beta prior
        beta                % Beta parameters of Beta prior
        pi                  % Posterior mean of Beta
        cutoff              % Threshold for variable inclusion
        acc_dict            % Map storing accuracy per iteration
        signal_entry_est    % Estimated signal variables
        n_iter              % Total iterations
        cluster_est         % Current cluster assignments
        entries_survived    % History of variable selections (n_iter x p)
        C                   % Current regularization/Bandit parameter
        C_min               % Minimum C for decay
        C_max               % Maximum C for decay
        C_history           % Store C value per iteration
    end

    methods
        %% Constructor
        function obj = sdp_kmeans_bandit_adaptiveC(X, K, C_min, C_max)
            obj.X = X;
            obj.K = K;
            [obj.p, obj.n] = size(X);
            obj.C_min = C_min;
            obj.C_max = C_max;
        end

        %% Initialization
        function set_bayesian_parameters(obj)
            obj.alpha = ones(1, obj.p);
            obj.beta  = ones(1, obj.p);
            obj.pi    = obj.alpha ./ (obj.alpha + obj.beta);
        end

        function initialize_cluster_est(obj)
            obj.acc_dict = containers.Map('KeyType', 'double', 'ValueType', 'double');
        end

        %% Main Loop
        function cluster_est_final = fit_predict(obj, n_iter, cluster_true)
            if nargin < 3, cluster_true = []; end

            obj.n_iter = n_iter;
            obj.set_bayesian_parameters();
            obj.initialize_cluster_est();
            obj.entries_survived = zeros(n_iter, obj.p);
            obj.C_history = zeros(1, n_iter);

            fprintf('Starting simulation: n=%d, p=%d, iter=%d\n', obj.n, obj.p, n_iter);

            for i = 1:n_iter
                % --- Adaptive C ---
                obj.C = obj.C_max - (obj.C_max - obj.C_min) * (i - 1) / (n_iter - 1);
                obj.cutoff = log(1 / obj.C) / log((1 + obj.C) / obj.C);
                obj.C_history(i) = obj.C;

                % 1. Choose
                variable_subset_now = obj.choose();
                obj.entries_survived(i, :) = variable_subset_now;

                % 2. Reward
                reward_now = obj.reward(variable_subset_now, i);

                % 3. Update
                obj.update(variable_subset_now, reward_now);

                % 4. Optional evaluation
                if ~isempty(cluster_true)
                    obj.evaluate_accuracy(obj.cluster_est, cluster_true, i);
                end

                if mod(i, 100) == 0
                    fprintf('Iteration %d: %d variables selected\n', i, sum(variable_subset_now));
                end
            end

            % Final clustering
            final_selection = obj.signal_entry_est;
            X_sub_final = obj.X(final_selection, :);
            obj.cluster_est = obj.get_cluster(X_sub_final, obj.K);

            if ~isempty(cluster_true)
                obj.evaluate_accuracy(obj.cluster_est, cluster_true, obj.n_iter + 1);
            end

            cluster_est_final = obj.cluster_est;
            fprintf('Total fit_predict done.\n');
        end

        %% Bandit logic
        function variable_subset = choose(obj)
            theta = betarnd(obj.alpha, obj.beta);
            variable_subset = theta > obj.cutoff;
        end

        function reward_vec = reward(obj, variable_subset, ~)
            X_sub = obj.X(variable_subset, :);
            obj.cluster_est = obj.get_cluster(X_sub, obj.K);

            reward_vec = zeros(1, obj.p);
            idx = find(variable_subset);

            if obj.K ~= 2
                warning('Reward function currently assumes K=2');
            end

            for j = 1:length(idx)
                feat_idx = idx(j);
                sample_c1 = X_sub(j, obj.cluster_est == 1);
                sample_c2 = X_sub(j, obj.cluster_est == 2);
                pval = permutationTest(sample_c1, sample_c2, 100);
                reward_vec(feat_idx) = pval < 0.05;
            end
        end

        function update(obj, variable_subset, reward_vec)
            obj.alpha(variable_subset) = obj.alpha(variable_subset) + reward_vec(variable_subset);
            obj.beta(variable_subset)  = obj.beta(variable_subset) + (1 - reward_vec(variable_subset));
            obj.pi = obj.alpha ./ (obj.alpha + obj.beta);
            obj.signal_entry_est = obj.pi > 0.5;
        end

        %% Clustering & Evaluation
        function cluster_est = get_cluster(~, X, K)
            cluster_est = get_cluster_by_sdp(X, K);
        end

        function evaluate_accuracy(obj, cluster_est, cluster_true, iter)
            obj.acc_dict(iter) = get_bicluster_accuracy(cluster_est, cluster_true);
        end

        %% Save Beta + C_history
        function save_beta_params(obj, filepath)
            alpha = obj.alpha;
            beta = obj.beta;
            C_history = obj.C_history;
            save(filepath, 'alpha', 'beta', 'C_history');
        end

        %% Save full results table
        function save_results(obj, output_dir, rep, Delta, support)
            if ~exist(output_dir, 'dir'), mkdir(output_dir); end
            results_table = obj.get_database_subtable(rep, Delta, support);
            fname = sprintf('res_adaptiveC_p%d_rep%d.mat', obj.p, rep);
            save(fullfile(output_dir, fname), 'results_table');
        end

        function database_subtable = get_database_subtable(obj, rep, Delta, support)
            [tp, fp, fn, ~] = obj.evaluate_discovery(support);
            n_row = obj.n_iter;
            acc_vec = cell2mat(values(obj.acc_dict))'; 

            database_subtable = table(...
                repelem(rep, n_row+1)', ...
                (1:(n_row+1))', ...
                repelem(Delta, n_row+1)', ...
                repelem(obj.p, n_row+1)', ...
                repelem(obj.n, n_row+1)', ...
                repelem(obj.C_max, n_row+1)', ...
                acc_vec, ...
                [0; tp], [0; fp], [0; fn], ...
                repelem(string(datetime('now')), n_row+1)', ...
                'VariableNames', ["rep", "iter", "sep", "dim", "n", "model", "acc", "true_pos", "false_pos", "false_neg", "jobdate"]);
        end

        function [tp, fp, fn, indices] = evaluate_discovery(obj, support)
            tp = zeros(obj.n_iter, 1); fp = zeros(obj.n_iter, 1); fn = zeros(obj.n_iter, 1);
            indices = strings(obj.n_iter, 1);
            for i = 1:obj.n_iter
                pos = obj.entries_survived(i, :);
                tp(i) = sum(pos(support));
                fp(i) = sum(pos) - tp(i);
                fn(i) = sum(~pos(support));
                indices(i) = strjoin(string(find(pos)), ',');
            end
        end
    end
end
