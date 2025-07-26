classdef data_generator_approximately_sparse_precision < data_generator_t
%% data_generator_approximately_sparse_precision
% @export
    methods
        function get_cov(obj, delta)
            obj.precision = get_precision_band(obj.p, 2, 0.45);
            obj.precision(obj.precision==0) =   delta;
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
 
