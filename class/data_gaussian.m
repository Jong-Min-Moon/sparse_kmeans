classdef data_gaussian
    properties
        dimension
        sample_size
        data
        sample_covariance
        entries_survived
        cluster_assign
        number_cluster
    end

    methods
        function obj = data_gaussian(data_matrix)
            obj.data = data_matrix;
            obj.dimension = size(obj.data,1);
            obj.sample_size = size(obj.data,2);
            obj.entries_survived = 1:obj.dimension;
            obj.number_cluster = 1;
        end
        
        function set_cluster_assign(cluster_assign)
            obj.cluster_assign = cluster_assign;
            obj.number_cluster = max(unique(cluster_assign));
        end
        
 
    end
end