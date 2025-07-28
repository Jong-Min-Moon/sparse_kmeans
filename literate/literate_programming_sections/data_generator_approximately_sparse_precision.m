classdef data_generator_approximately_sparse_precision < data_generator_t
%% data_generator_approximately_sparse_precision
% @export
    methods
        function get_cov(obj, delta)
            obj.precision = get_precision_band(obj.p, 2, 0.45);
for i = 1:obj.p - 2
    obj.precision(i, i+2) = delta;   % Upper second off-diagonal
    obj.precision(i+2, i) = delta;   % Lower second off-diagonal
end
            obj.Sigma = inv(obj.precision);
        end
    
 
        function [X,label] = get_data(obj, delta)
            obj.get_cov(delta);
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix();
             rng(obj.seed);
            X = mvnrnd(mean_matrix', obj.Sigma);
            X=X';
        end
    end % end of methods
end
 
