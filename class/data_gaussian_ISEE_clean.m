classdef data_gaussian_ISEE_clean < data_gaussian
methods
    function dg = data_gaussian_ISEE_clean(data_matrix, omega_sparsity)
        dg@data_gaussian(data_matrix, omega_sparsity); 
    end
    
end
methods (Access = protected)
    function innovated_data_mean = get_innovated_data(dg)

        [innovated_data_mean, innovated_data_noise, ~] = ISEE_bicluster(dg);
        dg.data_innovated = innovated_data_mean + innovated_data_noise;
    end
    function cutoff = get_cutoff(dg)
        lambda = sqrt(log(dg.dimension)/dg.sample_size);
        diverging_quantity = sqrt(log(dg.sample_size));
        cutoff = diverging_quantity*max(dg.omega_sparsity*lambda^2, lambda);
    end    

    function cluster_mean_innovated_mat = get_cluster_mean_innovated(dg)
        % innovated mean mat, clean version
        innovated_data_mean = dg.get_innovated_data();
        cluster_mean_innovated_mat = zeros(dg.dimension, dg.number_cluster);
        for i = 1:dg.number_cluster
            data_innovated_cluster = innovated_data_mean(:, (dg.cluster_assign ==  i));
            cluster_mean_innovated_mat(:,i) = mean(data_innovated_cluster, 2);
        end
    end

    
end


end

