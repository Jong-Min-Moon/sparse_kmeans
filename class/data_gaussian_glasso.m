classdef data_gaussian_glasso < data_gaussian_ISEE_dirty
methods
    function dgg = data_gaussian_glasso(data_matrix, omega_sparsity)
        dgg@data_gaussian_ISEE_dirty(data_matrix, omega_sparsity); 
    end
    
end
methods (Access = protected)
    function innovated_data_mean = get_innovated_data_mean(dgg)
        dgg.get_precision_matrix();
        dgg.data_innovated = dgg.Omega_hat*dgg.data;
        cluster_mean_innovated_mat = zeros(dgg.dimension, dgg.number_cluster);
        innovated_data_mean = dgg.data_innovated;
        end
    end

    function cutoff = get_cutoff(dgg)
            % need to generalize to K clusters. currently only works for two clusters.
            % also, currently only works for sigma = 1
            n_1 = sum(dgg.cluster_info_vec == 1);
            n_2 = sum(dgg.cluster_info_vec == 2);
            sample_size_multiplier = sqrt(1/n_1 + 1/n_2);
            cutoff = sample_size_multiplier * sqrt(2 * log(dgg.dimension)) * sqrt(dgg.Omega_diag_hat);
        end  

    
    function get_precision_matrix(dgg)
        [Omega_best, idx, score_vec] = glasso_bicluster(dgg, 30);
        dgg.Omega_hat = Omega_best;
        dgg.Omega_diag_hat = diag(Omega_best);
    end

    
end


end

