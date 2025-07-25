%% test_ER_isee_clean
% @export
function test_ER_isee_clean()
n = 500;
p = 400;
rep = 10;
addpath(genpath('/home1/jongminm/sparse_kmeans'));
% Set database and table
table_name = 'chime';
db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db';
% Model setup
model = 'chain45';
cluster_1_ratio = 0.5;
% Generate data
[data, label_true, mu1, mu2, sep, ~, beta_star]  = generate_gaussian_data(n, p, 4, model, rep, cluster_1_ratio);
% Run our method
cluster_estimte_isee = ISEE_kmeans_clean(data', 2, 30, true, 6, 5, 0.03);
% Evaluate clustering accuracy
acc = get_bicluster_accuracy(cluster_estimte_isee, label_true)
% Current timestamp for database
jobdate = datetime('now','Format','yyyy-MM-dd HH:mm:ss');
% Retry logic for database insertion
max_attempts = 10;
attempt = 1;
pause_time = 5;
while attempt <= max_attempts
    try
        % Open DB connection
        conn = sqlite(db_dir, 'connect');
        % Insert query
        insert_query = sprintf(['INSERT INTO %s (rep, sep, p, n, model, acc, jobdate) ' ...
                                'VALUES (%d, %.4f, %d, %d, "%s", %.6f, "%s")'], ...
                                table_name, rep, sep, p, n, model, acc, char(jobdate));
        % Execute insertion
        exec(conn, insert_query);
        close(conn);
        fprintf('Inserted result successfully on attempt %d.\n', attempt);
        break;
    catch ME
        if contains(ME.message, 'database is locked')
            fprintf('Database locked. Attempt %d/%d. Retrying in %d seconds...\n', ...
                    attempt, max_attempts, pause_time);
            pause(pause_time);
            attempt = attempt + 1;
        else
            rethrow(ME);
        end
    end
end
if attempt > max_attempts
    error('Failed to insert after %d attempts due to persistent database lock.', max_attempts);
end
end
%% 
% 
% 
% 
