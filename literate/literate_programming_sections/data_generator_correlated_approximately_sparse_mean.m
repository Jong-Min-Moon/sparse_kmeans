classdef data_generator_correlated_approximately_sparse_mean < data_generator_t_correlated
%% data_generator_correlated_approximately_sparse_mean
% @export
 
    methods
    
        function obj = data_generator_correlated_approximately_sparse_mean(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_t_correlated(n, p, s, sep, seed, cluster_1_ratio);
            
        end
        function mean_matrix = get_mean_matrix(obj, delta)
             beta  = obj.get_beta();
                    % Set class means
             mu1_primitive = beta;
             mu2_primitive = -beta;
             mu2_primitive(obj.s+1:end) = delta;
             mu1 = obj.precision \ mu1_primitive;
             mu2 = obj.precision \ mu2_primitive;
             % Create mean matrix
             mean_matrix = [repmat(mu1', obj.n1, 1); repmat(mu2', obj.n2, 1)];
             mean_matrix= mean_matrix';
        end
 
        function [X,label] = get_data(obj, sd, delta)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix(delta);
             noise_matrix = obj.get_noise_matrix(sd);
            X = noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
%% 
