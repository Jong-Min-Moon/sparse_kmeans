classdef data_gaussian < handle
    properties
        dimension
        sample_size
        data
        covariance_full
        entries_survived_boolean
        number_entries_survived
        cluster_assign
        number_cluster
        cluster_mean_mat
        cluster_mean_sparse_mat
    end

    methods
        function dg = data_gaussian(data_matrix)
            dg.data = data_matrix;
            dg.dimension = size(dg.data,1);
            dg.sample_size = size(dg.data,2);
            dg.entries_survived_boolean = repelem(true, dg.dimension);
            dg.number_entries_survived = sum(dg.entries_survived_boolean);
            dg.number_cluster = 1;
            dg.cluster_assign = repelem(1,dg.sample_size);
        end
        
        function set_cluster_assign(dg, cluster_assign)
            if length(cluster_assign) ~= dg.sample_size
                error("the number of the assignment vector must be the same as the sample size")
            end

            dg.cluster_assign = cluster_assign;
            dg.number_cluster = max(unique(dg.cluster_assign));
        end
        
        function get_cluster_mean(dg)
            dg.cluster_mean_mat = zeros(dg.dimension, dg.number_cluster);
            for i = 1:dg.number_cluster
                data_cluster = dg.data(:, (dg.cluster_assign ==  i));
                dg.cluster_mean_mat(:,i) = mean(data_cluster, 2);
            end
        end

        function get_cluster_mean_sparse(dg)
            dg.cluster_mean_sparse_mat = zeros(dg.number_entries_survived, dg.number_cluster);
            for i = 1:dg.number_cluster
                data_cluster_sparse = dg.data(dg.entries_survived_boolean, (dg.cluster_assign ==  i));
                dg.cluster_mean_sparse_mat(:,i) = mean(data_cluster_sparse, 2);
            end
        end
        
 
    end
end