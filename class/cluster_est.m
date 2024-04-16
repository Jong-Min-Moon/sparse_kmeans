classdef cluster_est < handle
    properties
        sample_size
        cluster_info_vec
        number_cluster
        cluster_partition
        cluster_info_string
    end
    
    methods
        function ce = cluster_est(cluster_info_vec)
            ce.sample_size = length(cluster_info_vec);
            full_index_vec = 1:ce.sample_size;
            ce.cluster_info_vec = cluster_info_vec;
            if size(ce.cluster_info_vec,2) == 1
                ce.cluster_info_vec  = ce.cluster_info_vec';
            end
            label_cluster = unique(cluster_info_vec);
            ce.number_cluster = length(label_cluster);

            % create a struct representation (which aligns with the paper)
            ce.cluster_partition = containers.Map( ...
                1:ce.number_cluster, ...
                repelem({ce.cluster_info_vec}, ce.number_cluster) ...
                );
            for i = 1:ce.number_cluster
                partition_now = full_index_vec(cluster_info_vec==i);
                ce.cluster_partition(i) = {partition_now};
            end % end of the for loop that creates the partition dictionary
        
            % create a string representation (for the database)
            ce.cluster_info_string = get_num2str_with_mark(ce.cluster_info_vec, ",");
        end %end of the constructor
    
        function acc_vec = evaluate_accuracy(ce, cluster_true)
            permutation_all = perms(1:ce.number_cluster);
            number_permutation = size(permutation_all, 1);
            acc_permutation_vec = zeros(number_permutation, 1);
            for j = 1:number_permutation
                permutation_now = permutation_all(j,:);
                cluster_permuted_now = ce.change_label(permutation_now);
                acc_permutation_vec(j) = mean(cluster_true == cluster_permuted_now);
            end % end of the for loop over permutations
            acc_vec = max( acc_permutation_vec );
        end % end of evaluate_accuracy

        function cluster_est_permuted = change_label(ce, permutation)
            cluster_est_permuted = ce.cluster_info_vec;
            for i = 1:ce.number_cluster
                cluster_est_permuted(ce.cluster_info_vec==i) = permutation(i);
            end
        end% end of change_label


    end% end of methods
end % end of the class