function merge_results_to_sqlite(mat_dir, db_path,
                                 table_name) files = dir(fullfile(mat_dir,
                                                                  'res_*.mat'));
conn = sqlite(db_path, 'connect');
fprintf('Merging %d files into %s...\n', length(files), table_name);

    for
      i = 1 : length(files) data = load(fullfile(mat_dir, files(i).name));
    % sqlwrite creates the table if it does not exist.%
        It appends if it does exist.sqlwrite(conn, table_name,
                                             data.results_table);
    if mod (i, 50)
      == 0, fprintf('Processed %d/%d\n', i, length(files));
    end end

        close(conn);
    fprintf('Success. Cleaning up .mat files...\n');
    % Optional : rmdir(mat_dir, 's');
    % Uncomment to delete temp files after merge end