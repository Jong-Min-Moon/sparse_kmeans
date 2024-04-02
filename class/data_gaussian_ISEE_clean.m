classdef data_gaussian_ISEE_clean < data_gaussian_ISEE_dirty
methods
    function dg = data_gaussian_ISEE_clean(data_matrix, omega_sparsity)
        dg@data_gaussian_ISEE_dirty(data_matrix, omega_sparsity); 
    end
    
end
methods (Access = protected)
    function innovated_data_mean = get_innovated_data_mean(dg)

        [innovated_data_mean, innovated_data_noise, ~] = ISEE_bicluster(dg);
        dg.data_innovated = innovated_data_mean + innovated_data_noise;
    end
    
    function cutoff = get_cutoff(dg)
        lambda = sqrt(log(dg.dimension)/dg.sample_size);
        diverging_quantity = sqrt(log(dg.sample_size));
        cutoff = diverging_quantity*max(dg.omega_sparsity*lambda^2, lambda);
    end    


    
end


end

