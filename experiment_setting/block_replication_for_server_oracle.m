classdef block_replication_for_server_oracle < handle
    properties
        table_name
        db_dir
        data_generator
        number_cluster
        n_iter_max
        window_size_half
        sample_size
        init_method
        cluster_true
        learner
        run_full
        matrix_sparsity
        data_obj
    end % end of properties

    methods
        function blfso = block_replication_for_server_oracle(table_name, db_dir, support, separation, dimension, correlation, sample_size, n_iter_max, run_full, init_method, matrix_sparsity)
            %save variables
            blfso.number_cluster = 2;
            blfso.n_iter_max = n_iter_max;
            blfso.window_size_half = 2;
            blfso.run_full = run_full;
            blfso.table_name     = table_name;
            blfso.db_dir         = db_dir;
            blfso.sample_size    = sample_size;
            blfso.init_method    = init_method;
            blfso.matrix_sparsity = matrix_sparsity;
            
            %model setting
            blfso.data_generator = sparse_symmetric_data_generator(support, separation, dimension, matrix_sparsity, correlation)
            blfso.data_obj = data_gaussian_oracle(data, omega_sparsity, blfso.data_generator.covariance_matrix, blfso.data_generator.sparse_precision_matrix);
            blfso.cluster_true = [repelem(1,sample_size/2), repelem(2,sample_size/2)];
        end % end of the constructer
        
        function database_subtable = run_one_replication(blfso, block_num, iter_num)
            % @data_gaussian_ISEE_clean
                rep = (block_num-1)*4+iter_num;
                fprintf("replication: (%i)th \n\n", rep)

                zero_mean = zeros(blfso.data_generator.dimension,1);
                x_noiseless = blfso.data_generator.get_noiseless_data(blfso.sample_size);
                

                rng(rep)
                x_noisy = x_noiseless +  mvnrnd(zero_mean, blfso.data_generator.covariance_matrix, blfso.sample_size)';%data generation. each column is one observation
                blfso.learner = iterative_kmeans(x_noisy, blfso.data_obj, blfso.number_cluster, blfso.data_generator.conditional_correlation, blfso.init_method);
                blfso.learner.run_iterative_algorithm(blfso.n_iter_max, blfso.window_size_half, 0.01, blfso.run_full);
    
                database_subtable = blfso.learner.get_database_subtable(rep, blfso.data_generator.separation, blfso.data_generator.conditional_correlation, blfso.data_generator.support, blfso.cluster_true, blfso.data_generator.sparse_precision_matrix);
        end%end of run_one_replication

        function run_four_replications(blfso, block_num)
            for iter_num = 1:4
                database_subtable = blfso.run_one_replication(blfso, block_num, iter_num);
                blfso.save_into_database(database_subtable);
            end % end of the for loop
        end % end of run_four_replications

        function save_into_database(blfso, database_subtable)
            random_seconds = randi([4 32],1);
            pause(random_seconds);
            conn=sqlite(blfso.db_dir);
            pause(2);
            try
                sqlwrite(conn, blfso.table_name, database_subtable)
            catch
                fprintf("db insertion failed")
            end
            pause(2);
            close(conn)
        end % end of save_into_database
    end % end of methods
end