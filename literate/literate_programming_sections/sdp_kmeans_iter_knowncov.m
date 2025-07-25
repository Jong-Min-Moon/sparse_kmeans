classdef sdp_kmeans_iter_knowncov < handle
%% sdp_kmeans_iter_knowncov
% @export
    properties
        X           % Data matrix (d x n)
        K           % Number of clusters
        n           % Number of data points
        p           % Data dimension
        cutoff      % Threshold for variable inclusion
        n_iter
  
    end
    methods
        function obj = sdp_kmeans_iter_knowncov(X, K)
            obj.X = X;
            obj.K = K;
            obj.n = size(X, 2);
            obj.p = size(X, 1);
            obj.n_iter = NaN;
            
            
            
        end
        
        function set_cutoff(obj)
            obj.cutoff = sqrt(2 * log(obj.p) );
        end
            
        function cluster_est_now = fit_predict(obj, n_iter)     
             % written 01/11/2024
             cluster_est_now = cluster_spectral(obj.X, obj.K); % initial clustering
             obj.set_cutoff();
 
            % iterate
            for iter = 1:n_iter
                fprintf("\n%i th iteration\n\n", iter)
                n_g1_now = sum(cluster_est_now == 1);
                n_g2_now = obj.n-n_g1_now;
                % 1. estimate cluster means
                if max(n_g1_now, n_g2_now) == obj.n
                    fprintf("all observations are clustered into one group")
                    cluster_est_now = repelem(1, obj.n);
                    return
                end
                
                % cluster 1 mean
                x_now_g1 = obj.X(:, (cluster_est_now ==  1));
                x_bar_g1 = mean(x_now_g1, 2);  
                % cluster 2 mean
                x_now_g2 = obj.X(:, (cluster_est_now ==  2));         
                x_bar_g2 = mean(x_now_g2, 2);
                % thresholding
                abs_diff = abs(x_bar_g1 - x_bar_g2) * sqrt( n_g1_now*n_g2_now/obj.n );
                thresholder_vec = abs_diff > obj.cutoff;
                fprintf("%i entries survived \n\n",sum(thresholder_vec))
                x_sub_now = obj.X(thresholder_vec,:);
                    % 3. apply SDP k-means   
                cluster_est_now = get_cluster_by_sdp(x_sub_now, obj.K); 
            end
        end % end of fit_predict
            
      
    end % end of methods
end
%% 
%% 
