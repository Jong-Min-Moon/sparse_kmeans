rho=rho/100
for jj = 1:4
    experimenter = block_iteration_for_server(table_name, db_dir, 1:10, separation, dimension, rho, sample_size, n_iter_max);
    database_subtable = experimenter.run_one_iteration(ii, jj)
    experimenter.save_into_database(database_subtable)
end


