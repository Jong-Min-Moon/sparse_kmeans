function merge_results_to_sqlite(mat_dir, db_path, table_name)
    % MERGE_RESULTS_TO_SQLITE Aggregates individual .mat results into SQLite
    
    % Find all result files following the naming convention
    files = dir(fullfile(mat_dir, 'res_*.mat'));
    
    if isempty(files)
        fprintf('No .mat files found in %s. Exiting.\n', mat_dir);
        return;
    end

    % Establish connection to the SQLite database
    conn = sqlite(db_path, 'connect');
    fprintf('Merging %d files into %s...\n', length(files), table_name);

    for i = 1:length(files)
        try
            % Load the individual result table
            data = load(fullfile(mat_dir, files(i).name));
            
            % Check if the expected variable exists in the file
            if isfield(data, 'results_table')
                % sqlwrite creates the table if it doesn't exist; appends if it does.
                sqlwrite(conn, table_name, data.results_table);
            else
                fprintf('Warning: "results_table" not found in %s\n', files(i).name);
            end
            
        catch ME
            fprintf('Error processing file %s: %s\n', files(i).name, ME.message);
        end

        % Progress tracker
        if mod(i, 50) == 0
            fprintf('Processed %d/%d files...\n', i, length(files));
        end
    end

    % Clean up
    close(conn);
    fprintf('Success. All data merged.\n');
    
    % To save disk space after a successful merge, uncomment the line below:
    % rmdir(mat_dir, 's'); 
end