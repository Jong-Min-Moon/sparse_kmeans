classdef block_replication_for_server_chime < handle
    properties
        table_name
        db_dir
        data_generator
        number_cluster
        n_iter_max
        window_size_half
        loop_detect_start
        sample_size
        init_method
        cluster_true
        learner
        run_full
        omega_sparsity
        data_obj
    end % end of properties
    methods
        function blfs = block_replication_for_server_chime(table_name, db_dir, support, separation, dimension, correlation, sample_size, n_iter_max, run_full, init_method, omega_sparsity, data_obj, flip, window_size_half, loop_detect_start)
            %save variables
            blfs.number_cluster = 2;
            blfs.n_iter_max = n_iter_max;
            blfs.window_size_half = window_size_half;
            blfs.loop_detect_start = loop_detect_start;
            blfs.run_full = run_full;
            blfs.table_name     = table_name;
            blfs.db_dir         = db_dir;
            blfs.sample_size    = sample_size;
            blfs.init_method    = init_method;
            blfs.omega_sparsity = omega_sparsity;
            blfs.data_obj = data_obj;
            
            %model setting
            blfs.data_generator = sparse_symmetric_data_generator(support, separation, dimension, omega_sparsity, correlation, flip)
            blfs.cluster_true = [repelem(1,sample_size/2), repelem(2,sample_size/2)];    
        end % end of the constructer
        function database_subtable = run_one_replication(blfs, block_num, iter_num)
                rep = (block_num-1)*20+iter_num;
                fprintf("replication: (%i)th \n\n", rep)

                zero_mean = zeros(blfs.data_generator.dimension,1);
                x_noiseless = blfs.data_generator.get_noiseless_data(blfs.sample_size);

                rng(rep)
                x_noisy = x_noiseless +  mvnrnd(zero_mean, blfs.data_generator.covariance_matrix, blfs.sample_size)';%data generation. each column is one observation
                %disp(blfs.data_generator.sparse_precision_matrix)
                if isstring(blfs.data_obj)
                    if strcmp(blfs.data_obj, "oracle")
                        data_obj_now = data_gaussian_oracle(x_noisy, blfs.omega_sparsity, blfs.data_generator.covariance_matrix, blfs.data_generator.sparse_precision_matrix);
                    end
                else
                    data_obj_now = blfs.data_obj(x_noisy, blfs.omega_sparsity);
                end
 
                n = blfs.sample_size;
                p = blfs.data_generator.dimension;
                s = blfs.data_generator.sparsity;               
                initialization_noise_scale = sqrt(s*log(p)/(n^2 * p));

                mu_1 = x_noiseless(:,1);
                mu_2 = x_noiseless(:,end);
                mu_mat = [mu_1, mu_2];
                beta_0 = (mu_1-mu_2);
                beta_0 = beta_0 + randn(1, p)' * initialization_noise_scale;

                

                blfs.learner = chime_simul(x_noisy, blfs.number_cluster);
                % = iterative_kmeans(data_obj_now, blfs.number_cluster, blfs.data_generator.conditional_correlation, blfs.init_method);
                blfs.learner.fit_predict( blfs.cluster_true, mu_mat, beta_0);
                
                database_subtable = blfs.learner.get_database_subtable(rep, blfs.data_generator.separation, blfs.data_generator.conditional_correlation, blfs.data_generator.support, blfs.cluster_true, blfs.data_generator.sparse_precision_matrix);
        end%end of run_one_replication


        function run_four_replications(blfs, block_num)
            for iter_num = 1:4
                database_subtable = blfs.run_one_replication(blfs, block_num, iter_num);
                blfs.save_into_database(database_subtable);
            end % end of the for loop
        end % end of run_four_replications

        function save_into_database(blfs, database_subtable)
            random_seconds = randi([4 32],1);
            pause(random_seconds);
            conn=sqlite(blfs.db_dir);
            pause(2);
            try
                sqlwrite(conn, blfs.table_name, database_subtable)
            catch
                fprintf("db insertion failed")
            end
            pause(2);
            close(conn)
        end % end of save_into_database
    end % end of methods
end
