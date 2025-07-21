%% insertTableIntoSQLite
% @export
function insertTableIntoSQLite(db_dir, table_name, obj, rep, Delta, support)
% insertTableIntoSQLite Inserts a MATLAB table into an SQLite database.
%
%   insertTableIntoSQLite(db_dir, table_name, obj, rep, Delta, support, max_attempts, pause_time)
%   generates a data table using obj.get_database_subtable and attempts to
%   insert it into the specified SQLite database table. It includes a retry
%   mechanism for database lock errors.
%
%   Inputs:
%     db_dir       - (char) Full path to the SQLite database file.
%     table_name   - (char) Name of the table within the database to insert into.
%     obj          - (object) An instance of a class (e.g., sdp_kmeans_bandit_simul)
%                    that has a method called 'get_database_subtable' and other
%                    properties needed by that method.
%     rep          - (numeric) Replication number for the simulation.
%     Delta        - (numeric) Separation parameter for the simulation.
%     support      - (array) Support vector for evaluating discovery.
%     max_attempts - (numeric) Maximum number of times to retry insertion
%                    if the database is locked.
%     pause_time   - (numeric) Time in seconds to pause between retries.
%
%   Example:
%     % Assuming 'myBanditObj', 'dbPath', 'tableName', 'repNum', 'deltaVal', 'supVec'
%     % are already defined and 'dbPath' exists.
%     % Also assume 'max_attempts' = 5 and 'pause_time' = 2
%     % insertTableIntoSQLite('my_database.db', 'simulation_results', myBanditObj, ...
%     %                       1, 0.5, [1 3 5], 5, 2);
max_attempts = 10;
pause_time=2;
% Input validation (basic checks)
if ~ischar(db_dir) || isempty(db_dir)
    error('insertTableIntoSQLite:InvalidDbDir', 'db_dir must be a non-empty character array (path to database).');
end
if ~ischar(table_name) || isempty(table_name)
    error('insertTableIntoSQLite:InvalidTableName', 'table_name must be a non-empty character array.');
end
if ~isobject(obj) || ~isprop(obj, 'n_iter') % Basic check if obj is a valid object
    error('insertTableIntoSQLite:InvalidObject', 'obj must be a valid object with required properties/methods.');
end
if ~ismethod(obj, 'get_database_subtable')
    error('insertTableIntoSQLite:MissingMethod', 'The provided object ''obj'' must have a method named ''get_database_subtable''.');
end
if ~isnumeric(rep) || ~isscalar(rep)
    error('insertTableIntoSQLite:InvalidRep', 'rep must be a numeric scalar.');
end
if ~isnumeric(Delta) || ~isscalar(Delta)
    error('insertTableIntoSQLite:InvalidDelta', 'Delta must be a numeric scalar.');
end
if ~isnumeric(support) || ~isvector(support)
    error('insertTableIntoSQLite:InvalidSupport', 'support must be a numeric vector.');
end
% Generate the table using the provided object's method
fprintf('Generating database subtable...\n');
try
    database_subtable = obj.get_database_subtable(rep, Delta, support);
catch ME
    error('insertTableIntoSQLite:TableGenerationError', 'Error generating database subtable: %s', ME.message);
end
if isempty(database_subtable) || ~istable(database_subtable)
    error('insertTableIntoSQLite:EmptyOrInvalidTable', 'The generated database_subtable is empty or not a valid MATLAB table.');
end
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
