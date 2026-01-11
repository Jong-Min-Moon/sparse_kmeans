%% insertTableIntoSQLite
% @export
function insertTableIntoSQLite(db_dir, table_name, database_subtable, rep, Delta, support)
max_attempts = 10;
pause_time=2;
 
attempt = 1;
while attempt <= max_attempts
    conn = []; % Initialize conn to ensure it's cleared if connection fails
    try
        % Attempt to connect to the database
        conn = sqlite(db_dir, 'connect');
        
        % *** THE CORRECT AND RECOMMENDED FIX: Use sqlwrite ***
        sqlwrite(conn, table_name, database_subtable); 
        
        % Close connection on success
        close(conn);
        fprintf('Inserted %d rows successfully into %s on attempt %d.\n', size(database_subtable, 1), table_name, attempt);
        return; % Exit the function on successful insertion
    catch ME
        % If connection was established, try to close it before retrying/rethrowing
        if ~isempty(conn) && isvalid(conn)
            close(conn); 
        end
        % Check if the error is due to database lock or busy status
        if contains(ME.message, 'database is locked', 'IgnoreCase', true) || ...
           contains(ME.message, 'SQLITE_BUSY', 'IgnoreCase', true)
            fprintf('Database locked. Attempt %d/%d. Retrying in %.1f seconds...\n', ...
                    attempt, max_attempts, pause_time);
            pause(pause_time);
            attempt = attempt + 1;
        else
            % If it's another error, rethrow it
            rethrow(ME);
        end
    end
end
% If the loop finishes without successful insertion
error('insertTableIntoSQLite:MaxAttemptsReached', 'Failed to insert after %d attempts due to persistent database lock.', max_attempts);
end
%% 
%% 
%% 
