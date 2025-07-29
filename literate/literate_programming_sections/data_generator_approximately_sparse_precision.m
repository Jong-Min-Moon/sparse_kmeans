classdef data_generator_approximately_sparse_precision < data_generator_t
%% data_generator_approximately_sparse_precision
% @export
    methods
        function get_cov(obj, delta)
           omat = get_precision_band(p, 2, 0.45);
           [mat, rn] = findPDMatrix(omat, delta);
           rn
           obj.precision = mat;
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
 
