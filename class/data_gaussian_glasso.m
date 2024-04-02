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
    end

    function cutoff = get_cutoff(dgg)
        lambda = sqrt(log(dgg.dimension)/dgg.sample_size);
        diverging_quantity = sqrt(log(dgg.sample_size));
        cutoff = diverging_quantity*max(dgg.omega_sparsity*lambda^2, lambda);
    end    

    
    function get_precision_matrix(dgg)
        [Omega_best, idx, score_vec] = glasso_bicluster(dgg, 30);
        dgg.Omega_hat = Omega_best;
        dgg.Omega_diag_hat = diag(Omega_best);
    end

    
end


end

