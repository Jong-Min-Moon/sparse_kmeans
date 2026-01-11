classdef sdp_kmeans_bandit < handle
%% sdp_kmeans_bandit
% @export
    properties
        X           % Data matrix (d x n)
        K           % Number of clusters
        n           % Number of data points
        p           % Data dimension
        cutoff      % Threshold for variable inclusion
        alpha       % Alpha parameters of Beta prior
        beta        % Beta parameters of Beta prior
        pi
        acc_dict
        signal_entry_est
        n_iter
        cluster_est
        entries_survived  
        C
    end
    methods
        function obj = sdp_kmeans_bandit(X, K, C)
            obj.X = X;
obj.K = K;
obj.n = size(X, 2);
obj.p = size(X, 1);
obj.C = C;
obj.cutoff = log(1 / C) / log((1 + C) / C);
obj.n_iter = NaN;
end

    function set_bayesian_parameters(obj) obj.alpha = ones(1, obj.p);
obj.beta = repmat(1, 1, obj.p);
obj.pi = obj.alpha./ (obj.alpha + obj.beta);
end

    function cluster_est_final =
        fit_predict(obj, n_iter, cluster_true) if nargin < 3 cluster_true = [];
end

    tic; % Start timing for the entire fit_predict method
            obj.n_iter = n_iter;
obj.set_bayesian_parameters();
obj.initialize_cluster_est();

% Pre - allocate entries_survived if possible obj.entries_survived =
    zeros(n_iter, obj.p);

            fprintf("initialization done\n")
            for i = 1:n_iter
                variable_subset_now = obj.choose();
            obj.entries_survived(i, :) = variable_subset_now;

            % Display reduced output disp([
              'number of arms pulled: ', mat2str(sum(variable_subset_now)), '\n'
            ]);

            reward_now = obj.reward(variable_subset_now, i);
            obj.update(variable_subset_now, reward_now);

            if
              ~isempty(cluster_true) obj.evaluate_accuracy(obj.cluster_est,
                                                           cluster_true, i);
            end end

                % final clustering final_selection = obj.signal_entry_est;
            X_sub_final = obj.X(final_selection, :);
            obj.cluster_est = obj.get_cluster(X_sub_final, obj.K);

            if
              ~isempty(cluster_true) obj.evaluate_accuracy(obj.cluster_est,
                                                           cluster_true,
                                                           obj.n_iter + 1);
            end

                cluster_est_final = obj.cluster_est;

            total_fit_predict_time = toc; % End timing for the entire fit_predict method            
            fprintf('Total fit_predict time: %.4f seconds\n', total_fit_predict_time);
            end

                function cluster_est =
                    get_cluster(obj, X, K) %
                    inherit this class and change this part to try simpler
                        clustering methods cluster_est =
                        get_cluster_by_sdp(X, K);
            end

                function initialize_cluster_est(obj) obj.acc_dict =
                containers.Map(1
                               : (obj.n_iter + 1), repelem(0, obj.n_iter + 1));
            end function variable_subset = choose(obj) theta =
                betarnd(obj.alpha, obj.beta);
            variable_subset = theta > obj.cutoff;
            end

                function reward_vec = reward(obj, variable_subset, iter) %
                                      Use only selected variables X_sub =
                                          obj.X(variable_subset,
                                                :);
            obj.cluster_est = obj.get_cluster(X_sub, obj.K);
            % Assume K = 2 sample_cluster_1 = X_sub( :, obj.cluster_est == 1);
            sample_cluster_2 = X_sub( :, obj.cluster_est == 2);
            % size(sample_cluster_1, 2) %
                size(sample_cluster_2, 2) reward_vec = zeros(1, obj.p);
            idx = find(variable_subset);
            % only calculate the p-values for selected variables
            for j = 1:length(idx)
                i = idx(j);
            pval = permutationTest(... sample_cluster_1(j,
                                                        :),
                                   ... sample_cluster_2(j,
                                                        :),
                                   ... 100 ...);
            % reward_vec(i) = pval < 0.05;
            end disp(
                [ 'number of rewarded pulls: ', mat2str(sum(reward_vec)) ]);
            end % end of method reward

                      function update(obj, variable_subset,
                                      reward_vec) obj.alpha(variable_subset) =
                obj.alpha(variable_subset) + reward_vec(variable_subset);
            obj.beta(variable_subset) = obj.beta(variable_subset) +
                                        (1 - reward_vec(variable_subset));
            obj.pi = obj.alpha./ (obj.alpha + obj.beta);
            obj.signal_entry_est = obj.pi > 0.5;
            end % end of method update

                      function evaluate_accuracy(obj, cluster_est, cluster_true,
                                                 iter) obj.acc_dict(iter) =
                get_bicluster_accuracy(cluster_est, cluster_true);
            obj.acc_dict(iter) end % end of method evaluate_accuracy

                function database_subtable = get_database_subtable(
                obj, rep, Delta, support) s = length(support);
            current_time = get_current_time();
            [ true_pos_vec, false_pos_vec, false_neg_vec, ~] =
                obj.evaluate_discovery(support);

            n_row = int32(obj.n_iter);
            database_subtable = table(
                ... repelem(
                    rep,
                    n_row +
                        1)',...                      % 01 replication number (1
                                                                              : (n_row +
                                                                                 1))',...                              % 02 step iteration number repelem(Delta,
                                                                                                                                                          n_row +
                                                                                                                                                              1)',...                    % 03 separation repelem(obj.p, n_row +
                                                                                                                                                                                                                            1)',...                    % 04 data dimension repelem(obj.n, n_row + 1)',...                      % 05 sample size repelem(obj.C,
                                                                                                                                                                                                                                                                                                                                                        n_row +
                                                                                                                                                                                                                                                                                                                                                            1)',...                        % 06 model ... cell2mat(values(obj.acc_dict))',...             % 07 accuracy ...
                    [0; true_pos_vec],
                ... % 10 true positive[0; false_pos_vec],
                ... % 11 false positive[0; false_neg_vec],
                ... %
                    12 false negative... repelem(
                        current_time,
                        n_row +
                            1)', ...            % 13 timestamp 'VariableNames',
                ...... % 1 2 3 4 5 6 ["rep", "iter", "sep", "dim", "n", "model",
                                      ...... % 7 "acc",
                                      ...... % 10 11 12 "true_pos", "false_pos",
                                      "false_neg", ...... 13 "jobdate"]);
            end % end of get_database_subtable

                      function[true_pos_vec, false_pos_vec, false_neg_vec,
                               survived_indices] =
                evaluate_discovery(obj, support) true_pos_vec =
                    zeros(obj.n_iter, 1);
            false_pos_vec = zeros(obj.n_iter, 1);
            false_neg_vec = zeros(obj.n_iter, 1);
            survived_indices = strings(obj.n_iter, 1);
            for
              i = 1 : obj.n_iter positive_vec = obj.entries_survived(i, :);
            true_pos_vec(i) = sum(positive_vec(support));
            false_pos_vec(i) = sum(positive_vec) - true_pos_vec(i);

            negative_vec = ~positive_vec;
            false_neg_vec(i) = sum(negative_vec(support));
            survived_indices(i) = get_num2str_with_mark(find(positive_vec),
                                                        ',');
            end end %
                end of evaluate_discovery

                    function save_results(obj, output_dir, rep, Delta,
                                          support) if ~exist(output_dir, 'dir'),
                mkdir(output_dir);
            end

                % Generate the table results_table =
                obj.get_database_subtable(rep, Delta, support);

            % Create a unique filename fname =
                sprintf('res_C%0.1f_p%d_rep%d.mat', obj.C, obj.p, rep);
            savepath = fullfile(output_dir, fname);

            save(savepath, 'results_table');
            fprintf('Saved results to %s\n', savepath);
            end end % end of methods end
