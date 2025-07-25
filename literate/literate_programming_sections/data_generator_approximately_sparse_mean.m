classdef data_generator_approximately_sparse_mean < data_generator_t
%% data_generator_approximately_sparse_mean
% @export
 
    methods
    
        function obj = data_generator_approximately_sparse_mean(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_t(n, p, s, sep, seed, cluster_1_ratio);
            
        end
        function noise_matrix = get_noise_matrix(obj, sd)
            % Generate noise once
            rng(obj.seed);
            noise_matrix = sd*normrnd(0,1,[obj.p, obj.n]);  % p x n1 noise
            empirical_sd = std(noise_matrix, 0, 'all');
            fprintf('--- empirical_sd =%f   ---\\n', empirical_sd);
            
        end
        function [X,label] = get_data(obj, sd, delta)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix();
            mean_matrix(obj.s+1:end, 1:obj.n1) =  delta; %approximate sparsity for cluster mean
            noise_matrix = obj.get_noise_matrix(sd);
            X = noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
%% 
