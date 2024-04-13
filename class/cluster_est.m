classdef cluster_est < handle
    properties
        sample_size
        cluster_info_vec
        number_cluster
        cluster_partition
    end
    
    methods
        function ce = cluster_est(cluster_info_vec)
            ce.sample_size = length(cluster_info_vec);
            full_index_vec = 1:ce.sample_size;
            ce.cluster_info_vec = cluster_info_vec;
            label_cluster = unique(cluster_info_vec);
            ce.number_cluster = length(label_cluster);
            ce.cluster_partition = dictionary( ...
                1:ce.number_cluster, ...
                repelem({ce.cluster_info_vec}, ce.number_cluster) ...
                );
            for i = 1:ce.number_cluster
                partition_now = full_index_vec(cluster_info_vec==i);
                ce.cluster_partition(i) = {partition_now};
            end % end of the for loop that creates the partition dictionary
        end %end of the constructor
    end

end % end of the class