classdef sdp_kmeans_iter_fixedsparsity < handle
    properties
        X           % Data matrix (d x n)
        K           % Number of clusters
        s           % Sparsity (number of features to select)
        n           % Number of data points
        p           % Data dimension
        time
        selected_features % Indices of features in final iteration
    end
    
    methods
        function obj = sdp_kmeans_iter_fixedsparsity(X, K, s)
            obj.X = X;
            obj.K = K;
            obj.s = s;
            obj.n = size(X, 2);
            obj.p = size(X, 1);
        end
        
        function cluster_est = get_initial_cluster(obj)
            % 1. Solve SDP on all features
            fprintf('Init: Solving full SDP on %d features...\n', obj.p);
            D_full = obj.X' * obj.X;
            Z_full = kmeans_sdp_pengwei(D_full, obj.K);
            
            % 2. Compute discriminating direction (Beta: Mean Difference)
            labels_full = sdp_sol_to_cluster(Z_full, obj.K);
            m1 = mean(obj.X(:, labels_full == 1), 2);
            m2 = mean(obj.X(:, labels_full == 2), 2);
            beta = m1 - m2;
            [~, sort_idx] = sort(abs(beta), 'descend');
            
            % 3. Keep top 5*s per user request
            n_init = min(obj.p, 5 * obj.s);
            subset = sort_idx(1:n_init);
            
            % 4. Solve SDP on subset to get initial labels
            fprintf('Init: Solving subset SDP on %d features...\n', length(subset));
            X_sub = obj.X(subset, :);
            D_sub = X_sub' * X_sub;
            Z_sub = kmeans_sdp_pengwei(D_sub, obj.K);
            cluster_est = sdp_sol_to_cluster(Z_sub, obj.K);
        end
        
        function [iternum, cluster_est_now] = fit_predict(obj, n_iter, loop_detect_start, window_size, min_delta)
            tic
            cluster_est_now = obj.get_initial_cluster();
            
            is_stop = 0;
            iternum = 0;
            rand_vec = nan(1, n_iter);
            
            while (~is_stop) && (iternum < n_iter)
                iternum = iternum + 1;
                fprintf("\n%i th iteration (Fixed Sparsity s=%d)\n", iternum, obj.s)
                
                n_g1 = sum(cluster_est_now == 1);
                n_g2 = obj.n - n_g1;
                
                if n_g1 == 0 || n_g2 == 0
                    fprintf("One cluster empty, stopping.\n")
                    return
                end
                
                % 1. Estimate means and differences (Beta)
                m1 = mean(obj.X(:, cluster_est_now == 1), 2);
                m2 = mean(obj.X(:, cluster_est_now == 2), 2);
                beta_now = m1 - m2;
                
                % 2. Select EXACTLY top s based on |beta|
                [~, sort_idx] = sort(abs(beta_now), 'descend');
                obj.selected_features = sort_idx(1:obj.s);
                
                % 3. Apply SDP K-means on the s features
                X_sub = obj.X(obj.selected_features, :);
                D_sub = X_sub' * X_sub;
                Z_new = kmeans_sdp_pengwei(D_sub, obj.K);
                cluster_est_new = sdp_sol_to_cluster(Z_new, obj.K);
                
                % 4. Stopping criteria
                rand_score = RandIndex(cluster_est_new, cluster_est_now);
                fprintf("Rand index with previous: %.4f\n", rand_score);
                rand_vec(iternum) = rand_score;
                
                is_stop = decide_stop_rand(rand_vec, loop_detect_start, window_size, min_delta);
                cluster_est_now = cluster_est_new;
            end
            obj.time = toc;
        end
    end
end
