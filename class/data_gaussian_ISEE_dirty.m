classdef data_gaussian_ISEE_dirty < data_gaussian
methods
    function dgd = data_gaussian_ISEE_dirty(data_matrix, omega_sparsity)
        dgd@data_gaussian(data_matrix, omega_sparsity); 
    end
    
end
methods (Access = protected)
    function innovated_data_mean_dirty = get_innovated_data_mean(dgd)

        [innovated_data_mean_clean, innovated_data_noise, Omega_diag_hat] = ISEE_bicluster(dgd);
        dgd.data_innovated = innovated_data_mean_clean + innovated_data_noise;
        innovated_data_mean_dirty = dgd.data_innovated;
        dgd.Omega_diag_hat = Omega_diag_hat;
    end

    function cutoff = get_cutoff(dgd)
        [GC,GR] = groupcounts(dgd.cluster_info_vec');
        cutoff = sqrt(2 * log(dgd.dimension)) / sqrt( prod(GC)/dgd.sample_size) * sqrt(dgd.Omega_diag_hat); %column vector
    end    

    function cluster_mean_innovated_mat = get_cluster_mean_innovated(dgd)
        % innovated mean mat, clean version
        innovated_data_mean = dgd.get_innovated_data_mean();
        cluster_mean_innovated_mat = zeros(dgd.dimension, dgd.number_cluster);
        
        %redundant procedure, just for unity with the dirty version
        for i = 1:dgd.number_cluster
            data_innovated_cluster = innovated_data_mean(:, (dgd.cluster_info_vec ==  i));
            cluster_mean_innovated_mat(:,i) = mean(data_innovated_cluster, 2);
        end
    end


    
end


end

