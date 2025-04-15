classdef block_replication_for_server < handle
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
        function blfs = block_replication_for_server(table_name, db_dir, support, separation, dimension, correlation, sample_size, n_iter_max, run_full, init_method, omega_sparsity, data_obj, flip, window_size_half, loop_detect_start)
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