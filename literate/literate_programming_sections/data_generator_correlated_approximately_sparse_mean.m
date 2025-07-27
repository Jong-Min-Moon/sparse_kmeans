classdef data_generator_correlated_approximately_sparse_mean < data_generator_t_correlated
%% data_generator_correlated_approximately_sparse_mean
% @export
 
    methods
    
        function obj = data_generator_correlated_approximately_sparse_mean(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_t_correlated(n, p, s, sep, seed, cluster_1_ratio);
            
        end
        function mean_matrix = get_mean_matrix(obj, delta)
 
             mu1_primitive = obj.get_beta();
             mu2_primitive = -mu1_primitive;
             n_delta = floor(0.1*obj.p);
             mu2_primitive(obj.s+1:n_delta) = delta;
             mu1 = obj.precision \ mu1_primitive;
             mu2 = obj.precision \ mu2_primitive;
             % Create mean matrix
             mean_matrix = [repmat(mu1', obj.n1, 1); repmat(mu2', obj.n2, 1)];
             mean_matrix= mean_matrix';
             obj.mean_matrix = mean_matrix;
        end
       function noise_matrix = get_noise_matrix(obj) %modification: t noise -> Gaussian noise
            % Generate noise once
            rng(obj.seed);
            noise_matrix = mvnrnd(zeros([obj.n, obj.p]), obj.Sigma); %$Gaussian noise
            noise_matrix = noise_matrix'; % p x n matrix
            obj.noise_matrix = noise_matrix;
 
        end        
        function [X,label] = get_data(obj, delta)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix(delta);
            noise_matrix = obj.get_noise_matrix();
            X = noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
%% 
