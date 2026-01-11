classdef data_generator_correlated_different_cov < data_generator_correlated_approximately_sparse_mean
%% data_generator_correlated_different_cov
% @export
 
    methods
    
        function obj = data_generator_correlated_different_cov(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_correlated_approximately_sparse_mean(n, p, s, sep, seed, cluster_1_ratio);
        end
 
        function [X,label] = get_data(obj, delta)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix(0);
            noise_matrix = obj.get_noise_matrix(delta);
            noise_matrix(:, 1:obj.n1      ) = (1+delta)*noise_matrix(:,1:obj.n1);
            noise_matrix(:, (obj.n1+1) : n) = (1-delta)*noise_matrix(:, (obj.n1+1) : n);
            X = noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
%% 
%% 
