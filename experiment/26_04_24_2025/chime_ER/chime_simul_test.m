n = 200;
p = 800;
rep = 1;

addpath(genpath('/home1/jongminm/sparse_kmeans'));

% Set database and table
table_name = 'chime';
db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db';

% Model setup
model = 'ER';
cluster_1_ratio = 0.5;

% Generate data
[data, label_true, mu1, mu2, ~, beta_star] = generate_gaussian_data(n, p, model, rep, cluster_1_ratio);
data = data';  % for CHIME, data should be n x p
true_cluster_mean = [mu1 mu2];
lambda_multiplier = [0.1 0.5 1 2 4 8 16];

% Run CHIME
[~, ~, ~, ~, ~, ~, ~, cluster_est] = CHIME(data, data, label_true, cluster_1_ratio, true_cluster_mean, beta_star, 0.1, lambda_multiplier, 100);

% Evaluate clustering accuracy
acc = get_bicluster_accuracy(cluster_est, label_true);

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
