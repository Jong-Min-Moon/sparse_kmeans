classdef block_iteration_for_server < handle
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
    end % end of properties

    methods
        function blfs = block_iteration_for_server(table_name, db_dir, support, separation, dimension, correlation, sample_size, n_iter_max)
            blfs.number_cluster = 2;
            blfs.n_iter_max = n_iter_max;
            blfs.window_size_half = 2;
            blfs.table_name     = table_name;
            blfs.db_dir         = db_dir;
            blfs.sample_size    = sample_size;
            blfs.data_generator = sparse_symmetric_data_generator(support, separation, dimension, 2, correlation)
            blfs.init_method    = 'spec';
            blfs.cluster_true = [repelem(1,sample_size/2), repelem(2,sample_size/2)];
        end % end of the constructer
        
        function database_subtable = run_one_iteration(blfs, block_num, iter_num)
            % @data_gaussian_ISEE_clean
                rep = (block_num-1)*4+iter_num;
                rng(rep)
                zero_mean = zeros(blfs.data_generator.dimension,1);
                x_noiseless = blfs.data_generator.get_noiseless_data(blfs.sample_size);
                x_noisy = x_noiseless +  mvnrnd(zero_mean, blfs.data_generator.covariance_matrix, blfs.sample_size)';%data generation. each column is one observation

                fprintf("replication: (%i)th \n\n", rep)
                blfs.learner = iterative_kmeans(x_noisy, @data_gaussian_ISEE_clean, blfs.number_cluster, blfs.data_generator.conditional_correlation, blfs.init_method);
                blfs.learner.run_iterative_algorithm(blfs.n_iter_max, blfs.window_size_half, 0.01);
    
                database_subtable = blfs.learner.get_database_subtable(rep, blfs.data_generator.separation, blfs.data_generator.conditional_correlation, blfs.data_generator.support, blfs.cluster_true, blfs.data_generator.sparse_precision_matrix);
        end%end of run_one_iteration

        function run_four_iterations(blfs, block_num)
            for iter_num = 1:4
                database_subtable = blfs.run_one_iteration(blfs, block_num, iter_num);
                blfs.save_into_database(database_subtable);
            end % end of the for loop
        end % end of run_four_iterations

        function save_into_database(blfs, database_subtable)
            random_seconds = randi([4 32],1);
            pause(random_seconds);
            conn=sqlite(blfs.db_dir);
            pause(2);
            sqlwrite(conn, blfs.table_name, database_subtable);
            pause(2);
            close(conn)
        end % end of save_into_database
    end % end of methods
end