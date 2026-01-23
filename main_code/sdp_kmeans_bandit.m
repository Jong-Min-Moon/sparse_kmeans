classdef sdp_kmeans_bandit < handle
    % SDP_KMEANS_BANDIT A Thompson Sampling approach to Sparse K-Means
    
    properties
        X                   % Data matrix (d x n)
        K                   % Number of clusters
        n                   % Number of data points
        p                   % Data dimension
        cutoff              % Threshold for variable inclusion
        alpha               % Alpha parameters of Beta prior
        beta                % Beta parameters of Beta prior
        pi                  % Probability of inclusion (posterior mean)
        acc_dict            % Map storing accuracy per iteration
        signal_entry_est    % Estimated signal variables
        n_iter              % Total iterations
        cluster_est         % Current cluster assignments
        entries_survived    % History of variable selections (n_iter x p)
        C                   % Regularization/Bandit parameter
    end

    methods
        %% Constructor
        function obj = sdp_kmeans_bandit(X, K, C)
            obj.X = X;
            obj.K = K;
            [obj.p, obj.n] = size(X);
            obj.C = C;
            % Precompute cutoff based on cost parameter C
            obj.cutoff = log(1 / C) / log((1 + C) / C);
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

        %% Main Simulation Loop
        function cluster_est_final = fit_predict(obj, n_iter, cluster_true)
            if nargin < 3, cluster_true = []; end
            
            tic;
            obj.n_iter = n_iter;
            obj.set_bayesian_parameters();
            obj.initialize_cluster_est();
            obj.entries_survived = zeros(n_iter, obj.p);

            fprintf('Starting simulation: n=%d, p=%d, iter=%d\n', obj.n, obj.p, n_iter);

            for i = 1:n_iter
                % 1. Selection step (Thompson Sampling)
                variable_subset_now = obj.choose();
                obj.entries_survived(i, :) = variable_subset_now;

                % 2. Reward step
                reward_now = obj.reward(variable_subset_now, i);
                
                % 3. Update Step
                obj.update(variable_subset_now, reward_now);

                % 4. Evaluation (Optional)
                if ~isempty(cluster_true)
                    obj.evaluate_accuracy(obj.cluster_est, cluster_true, i);
                end
                
                if mod(i, 100) == 0
                    fprintf('Iteration %d: %d arms pulled\n', i, sum(variable_subset_now));
                end
            end

            % Final Clustering using stabilized signal estimates
            final_selection = obj.signal_entry_est;
            X_sub_final = obj.X(final_selection, :);
            obj.cluster_est = obj.get_cluster(X_sub_final, obj.K);

            if ~isempty(cluster_true)
                obj.evaluate_accuracy(obj.cluster_est, cluster_true, obj.n_iter + 1);
            end

            cluster_est_final = obj.cluster_est;
            fprintf('Total fit_predict time: %.4f seconds\n', toc);
        end

        %% Bandit Logic
        function variable_subset = choose(obj)
            % Draw from Beta distribution and apply cutoff
            theta = betarnd(obj.alpha, obj.beta);
            variable_subset = theta > obj.cutoff;
        end

        function reward_vec = reward(obj, variable_subset, ~)
            X_sub = obj.X(variable_subset, :);
            obj.cluster_est = obj.get_cluster(X_sub, obj.K);
            
            % Split data for testing (Assumes K=2)
            sample_c1 = X_sub(:, obj.cluster_est == 1);
            sample_c2 = X_sub(:, obj.cluster_est == 2);
            
            reward_vec = zeros(1, obj.p);
            idx = find(variable_subset);
            
            % Conduct p-value tests on selected dimensions only
            for j = 1:length(idx)
                feat_idx = idx(j);
                % Simplified logic: Reward if feature separates clusters
                pval = permutationTest(sample_c1(j,:), sample_c2(j,:), 100);
                reward_vec(feat_idx) = pval < 0.05;
            end
        end

        function update(obj, variable_subset, reward_vec)
            % Bayesian update of the Beta distribution
            obj.alpha(variable_subset) = obj.alpha(variable_subset) + reward_vec(variable_subset);
            obj.beta(variable_subset)  = obj.beta(variable_subset)  + (1 - reward_vec(variable_subset));
            
            % Update posterior mean and hard-selection estimate
            obj.pi = obj.alpha ./ (obj.alpha + obj.beta);
            obj.signal_entry_est = obj.pi > 0.5;
        end

        %% Clustering & Evaluation
        function cluster_est = get_cluster(~, X, K)
            % Placeholder for SDP or alternate clustering
            cluster_est = get_cluster_by_sdp(X, K);
        end

        function evaluate_accuracy(obj, cluster_est, cluster_true, iter)
            obj.acc_dict(iter) = get_bicluster_accuracy(cluster_est, cluster_true);
        end

        %% Data Export
        function database_subtable = get_database_subtable(obj, rep, Delta, support)
            [tp, fp, fn, ~] = obj.evaluate_discovery(support);
            n_row = obj.n_iter;
            
            % Align accuracy dictionary values into vector
            acc_vec = cell2mat(values(obj.acc_dict))'; 

            database_subtable = table(...
                repelem(rep, n_row+1)', ...
                (1:(n_row+1))', ...
                repelem(Delta, n_row+1)', ...
                repelem(obj.p, n_row+1)', ...
                repelem(obj.n, n_row+1)', ...
                repelem(obj.C, n_row+1)', ...
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

function save_results(obj, output_dir, rep, Delta, support)
    % Ensure the directory exists
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Generate the standard results table
    results_table = obj.get_database_subtable(rep, Delta, support);

    % Prepare filename
    fname = sprintf('res_C%0.1f_p%d_rep%d.mat', obj.C, obj.p, rep);

    % Save results table AND final Beta parameters
    alpha_final = obj.alpha;
    beta_final  = obj.beta;
    pi_final    = obj.pi;

    save(fullfile(output_dir, fname), 'results_table', 'alpha_final', 'beta_final', 'pi_final');
end
    end
end