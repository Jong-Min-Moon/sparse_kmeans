classdef sparse_symmetric_data_generator < handle 
    properties
        separation
        support
        sparsity
        magnitude
        dimension
        precision_sparsity
        conditional_correlation
        sparse_precision_matrix
        covariance_matrix
    end

    methods
        function sdg = sparse_symmetric_data_generator(support, separation, dimension, precision_sparsity, conditional_correlation)
            sdg.support = support;
            sdg.sparsity = length(support);
            sdg.separation = separation;
            sdg.dimension = dimension;
            sdg.precision_sparsity = precision_sparsity;
            sdg.conditional_correlation = conditional_correlation;
            sdg.sparse_precision_matrix = sdg.get_sparse_precision_matrix();
            sdg.covariance_matrix = inv(sdg.sparse_precision_matrix);
            sdg.get_magnitude();

        end
        
        function [mean_1, mean_2] = get_symmetric_mean_sparse_before_innovation(sdg)
            sparse_pre_mean_one = sdg.get_sparse_mean_one();
            sparse_pre_mean_0 =  sdg.magnitude * sparse_pre_mean_one;
            mean_0 = sdg.covariance_matrix*sparse_pre_mean_0; % not s-sparse
            mean_1 = -mean_0; % not s-sparse
            mean_2 = mean_0;  % not s-sparse
        end

        function sparse_mean_one = get_sparse_mean_one(sdg)
            sparse_mean_one = zeros(sdg.dimension,1);
            sparse_mean_one(sdg.support) = 1;
        end

        function get_magnitude(sdg)
            %for symmetric_mean_sparse_before_innovation
            sdg.magnitude = sdg.separation/2/ sqrt( sum( sdg.covariance_matrix(sdg.support, sdg.support),"all") );
        end
        
        function sparse_precision_matrix = get_sparse_precision_matrix(sdg)
            sparse_precision_matrix = eye(sdg.dimension);
            for i = 1 : floor(sdg.precision_sparsity/2)
                off_diag_up = diag(sdg.conditional_correlation*ones(sdg.dimension-i,1), i);
                off_diag_low = diag(sdg.conditional_correlation*ones(sdg.dimension-i,1), -i);
                sparse_precision_matrix = sparse_precision_matrix + off_diag_up + off_diag_low;
            end
        end

        
    end% end of methods

end
