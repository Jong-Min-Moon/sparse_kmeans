classdef block_replication_for_server_bandit < handle
    methods
        function database_subtable = run_one_replication(blfs, block_num, iter_num)
                rep = (block_num-1)*4+iter_num;
                fprintf("replication: (%i)th \n\n", rep)

                zero_mean = zeros(blfs.data_generator.dimension,1);
                x_noiseless = blfs.data_generator.get_noiseless_data(blfs.sample_size);

                rng(rep)
                x_noisy = x_noiseless +  mvnrnd(zero_mean, blfs.data_generator.covariance_matrix, blfs.sample_size)';%data generation. each column is one observation
                if isstring(blfs.data_obj)
                    if strcmp(blfs.data_obj, "oracle")
                        data_obj_now = data_gaussian_oracle(x_noisy, blfs.omega_sparsity, blfs.data_generator.covariance_matrix, blfs.data_generator.sparse_precision_matrix);
                    end
                else
                    data_obj_now = blfs.data_obj(x_noisy, blfs.omega_sparsity);
                end
                blfs.learner = sdp_kmeans_bandit_simul(data_obj_now, blfs.number_cluster);
                % = iterative_kmeans(data_obj_now, blfs.number_cluster, blfs.data_generator.conditional_correlation, blfs.init_method);
                blfs.learner.run_iterative_algorithm(blfs.n_iter_max, blfs.cluster_true);
    
                database_subtable = blfs.learner.get_database_subtable(rep, blfs.data_generator.separation, blfs.data_generator.conditional_correlation, blfs.data_generator.support, blfs.cluster_true, blfs.data_generator.sparse_precision_matrix);
        end%end of run_one_replication
    end % end of methods
end
