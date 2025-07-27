classdef data_generator_correlated_different_cov < data_generator_t_correlated
%% data_generator_correlated_different_cov
% @export
 
    methods
    
        function obj = data_generator_correlated_different_cov(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_t_correlated(n, p, s, sep, seed, cluster_1_ratio);
        end
 
     
        function [X,label] = get_data(obj,   sd, delta)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix();
            noise_matrix = mvnrnd(mean_matrix', obj.Sigma); %$Gaussian noise
            noise_matrix = noise_matrix';
            noise_matrix(:,1:obj.n1)         = sd* (1+delta) * noise_matrix(:,1:obj.n1);
            noise_matrix(:,(obj.n1+1):obj.n) = sd * (1-delta) * noise_matrix(:,(obj.n1+1):obj.n);
            X = sqrtm(obj.Sigma) * noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
