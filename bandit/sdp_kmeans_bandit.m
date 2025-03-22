classdef sdp_kmeans_bandit < handle

    properties
        X           % Data matrix (d x n)
        K           % Number of clusters
        n           % Number of data points
        p           % Data dimension
        cutoff      % Threshold for variable inclusion
        alpha       % Alpha parameters of Beta prior
        beta        % Beta parameters of Beta prior
        pi
    end

    methods
        function obj = sdp_kmeans_bandit(X, K)
            % Constructor
            if nargin < 2
                error('Two input arguments required: data matrix X and number of clusters K.');
            end
            if ~ismatrix(X) || ~isnumeric(X)
                error('Input X must be a numeric matrix.');
            end
            if ~isscalar(K) || K <= 1 || K ~= floor(K)
                error('Number of clusters K must be an integer greater than 1.');
            end

            obj.X = X;
            obj.K = K;
            obj.n = size(X, 2)
            obj.p = size(X, 1)

            C = 0.5;
            obj.cutoff = log(1 / C) / log((1 + C) / C);

            obj.alpha = ones(1, obj.p);
            obj.beta = repmat(1, 1, obj.p);
            obj.pi = obj.alpha ./ (obj.alpha + obj.beta);
        end
        
   
        function fit_predict(obj, n_iter)
            for i = 1:n_iter
                variable_subset_now = obj.choose();
                disp(['Iteration ', num2str(i), ' - Chosen variables: ', mat2str(find(variable_subset_now))]);
                reward_now = obj.reward(variable_subset_now);
                obj.update(variable_subset_now, reward_now)
            end
        end

        function variable_subset = choose(obj)
            theta = betarnd(obj.alpha, obj.beta);
            variable_subset = theta > obj.cutoff;
        end
        
        function reward_vec = reward(obj, variable_subset)
            % Use only selected variables
            X_sub = obj.X(variable_subset, :);
            kmeans_learner = sdp_kmeans(X_sub, obj.K);
            cluster_est = kmeans_learner.fit_predict();

            % Assume K = 2
            sample_cluster_1 = X_sub(:, cluster_est == 1);
            sample_cluster_2 = X_sub(:, cluster_est == 2);
            %size(sample_cluster_1, 2)
            %size(sample_cluster_2, 2)

            reward_vec = zeros(1, obj.p);
            idx = find(variable_subset);

            % only calculate the p-values for selected variables
            for j = 1:length(idx)
                i = idx(j);
                p_val =  permutationTest( ...
                    sample_cluster_1(j, :), ...
                    sample_cluster_2(j, :), ...
                    100 ...
                ); % 
                reward_vec(i) = p_val <0.1;
            end
            reward_vec(11)           
        end % end of method reward

        function update(obj, variable_subset, reward_vec)
            obj.alpha(variable_subset) = obj.alpha(variable_subset) + reward_vec(variable_subset);
            obj.beta(variable_subset) = obj.beta(variable_subset) + (1 - reward_vec(variable_subset));
            obj.pi = obj.alpha ./ (obj.alpha + obj.beta);            
    end
end
