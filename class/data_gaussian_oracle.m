classdef data_gaussian_oracle < data_gaussian
    %original data container. Data here does not change over iteration
    properties
        %in addition to inherited properties,
        covariance
        precision
    end

    methods
        function dgo = data_gaussian_oracle(data, omega_sparsity, covariance, precision)
            dgo@data_gaussian(data, omega_sparsity);
            dgo.covariance = covariance;
            dgo.precision = precision;
            dgo.data_innovated = dgo.precision * dgo.data;
        end
    end

 

    
    methods (Access = protected)
        function cutoff = get_cutoff(dgo)
            % need to generalize to K clusters. currently only works for two clusters.
            % also, currently only works for sigma = 1
            n_1 = sum(dgo.cluster_info_vec == 1);
            n_2 = sum(dgo.cluster_info_vec == 2);
            sample_size_multiplier = sqrt(1/n_1 + 1/n_2);
            cutoff = sample_size_multiplier * sqrt(2 * log(dgo.dimension)) * sqrt(diag(dgo.precision));
        end
        
  




        function covariance_small = get_sample_covariance_small(dgo)
            %oracle
            covariance_small = dgo.covariance(dgo.support,dgo.support);
        end



    end %end of hidden method
end