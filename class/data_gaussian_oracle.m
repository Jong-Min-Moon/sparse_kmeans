classdef data_gaussian_oracle < data_gaussian
    %original data container. Data here does not change over iteration
    properties
        dimension
        sample_size
        data
        affinity
        sparse_affinity
        data_innovated
        Omega_hat
        Omega_diag_hat
        support
        number_support
        cluster_info_vec
        number_cluster
        cluster_mean_small_mat
        omega_sparsity
        covariance
        precision
    end

    methods
        function dgo = data_gaussian_oracle(data, omega_sparsity, covariance, precision)
            dgo@data_gaussian(data, omega_sparsity);
            dgo.covariance = covariance;
            dgo.precision = precision;
            dgo.data_innovated = dgo.precision * dg.data;
        end
    end

 

    
    methods (Access = protected)
        function cutoff = get_cutoff(dgo)
            % need to generalize to K clusters. currently only works for two clusters.
            % also, currently only works for sigma = 1
            n_1 = sum(dgo.cluster_info_vec == 1);
            n_2 = sum(dgo.cluster_info_vec == 2);
            sample_size_multiplier = sqrt(1/n_1 + 1/n_2);
            cutoff = sample_size_multiplier * sqrt(2 * log(dg.dimension)) * (1./diag(dgo.precision));
        end
        
  




        function covariance_small = get_sample_covariance_small(dgo)
            %oracle
            covariance_small = dgo.covariance(dgo.support,dgo.support);
        end



    end %end of hidden method
end