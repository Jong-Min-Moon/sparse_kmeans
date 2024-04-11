classdef data_gaussian < handle
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
        cluster_assign
        number_cluster
        cluster_mean_small_mat
        omega_sparsity
    end

    methods
        function dg = data_gaussian(data, omega_sparsity)
            dg.data = data;
            dg.dimension = size(dg.data,1);
            dg.sample_size = size(dg.data,2);
            dg.affinity = data'*data;
            
            dg.support = repelem(true, dg.dimension);
            dg.number_support = sum(dg.support);
            dg.number_cluster = 1;
            dg.cluster_assign = repelem(1,dg.sample_size);
            dg.omega_sparsity = omega_sparsity;
        end
    end

    methods
        function [data_innovated_small, data_innovated_big, sample_covariance_small] = threshold(dg, cluster_assign, omega_sparsity)
            if length(cluster_assign) ~= dg.sample_size
                error("the number of the assignment vector must be the same as the sample size")
            end
            
            dg.cluster_assign = cluster_assign;
            dg.number_cluster = max(unique(dg.cluster_assign));
            

            entrywise_signal_estimate = dg.get_entrywise_signal();
            dg.get_support(entrywise_signal_estimate);
            
            
            data_innovated_small = dg.data_innovated(dg.support,:);
            data_innovated_big = dg.data_innovated;
            sample_covariance_small = dg.get_sample_covariance_small();
        end  
    end

    
    methods (Access = protected)
        function entrywise_signal_estimate = get_entrywise_signal(dg)
            cluster_mean_innovated_mat = dg.get_cluster_mean_innovated();
            number_combination = nchoosek(dg.number_cluster,2);
            signal_estimate = zeros(dg.dimension, number_combination);
            for i = 1:(dg.number_cluster-1)
                for j = i+1:dg.number_cluster
                    column_index = i+j-2;
                    signal_estimate(:,column_index) = abs(cluster_mean_innovated_mat(:,i) - cluster_mean_innovated_mat(:,j));
                end
            end
            entrywise_signal_estimate = min(signal_estimate, [],2); %column vector
        end

        function cluster_mean_innovated_mat = get_cluster_mean_innovated(dg)
            dg.get_innovated_data();
            cluster_mean_innovated_mat = zeros(dg.dimension, dg.number_cluster);
            for i = 1:dg.number_cluster
                data_innovated_cluster = dg.data_innovated(:, (dg.cluster_assign ==  i));
                cluster_mean_innovated_mat(:,i) = mean(data_innovated_cluster, 2);
            end
        end


        function get_support(dg, entrywise_signal_estimate)
            cutoff = dg.get_cutoff();
            dg.support = entrywise_signal_estimate > cutoff;
            dg.number_support = sum(dg.support);
            dg.get_sparse_affinity()
        end

        function cutoff = get_cutoff(dg)
            cutoff = sqrt(2 * log(dg.dimension));
        end
        
        function sparse_affinity = get_sparse_affinity(dg)
            data_small = dg.data(dg.support, :);
            dg.sparse_affinity = data_small' * data_small;
        end







 


        

        function get_cluster_mean_small(dg)
            dg.cluster_mean_small_mat = zeros(dg.number_support, dg.number_cluster);
            for i = 1:dg.number_cluster
                data_cluster_small = dg.data(dg.support, (dg.cluster_assign ==  i));
                dg.cluster_mean_small_mat(:,i) = mean(data_cluster_small, 2);
            end
        end



        function sample_covariance_small = get_sample_covariance_small(dg)
            dg.get_cluster_mean_small();
            data_small = dg.data(dg.support,:);
            for i = 1:dg.number_cluster
                data_small(:,dg.cluster_assign == i) = data_small(:,dg.cluster_assign == i) - dg.cluster_mean_small_mat(:,i);
            end
            sample_covariance_small = (data_small * data_small')/(dg.sample_size-1);
        end



    end %end of hidden method
end