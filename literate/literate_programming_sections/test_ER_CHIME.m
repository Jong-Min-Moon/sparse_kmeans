%% test_ER_CHIME
% @export
function test_ER_CHIME()
n = 200;
p = 1000;
rep = 1;
addpath(genpath('/home1/jongminm/sparse_kmeans'));
% Set database and table
table_name = 'chime';
db_dir = '/home1/jongminm/sparse_kmeans/sparse_kmeans.db';
% Model setup
model = 'ER';
cluster_1_ratio = 0.5;
[data, label_true, mu1, mu2, sep, ~, beta_star]  = generate_gaussian_data(n, p, 4, model, rep, cluster_1_ratio);
noise_std = 1/10;
true_cluster_mean = [mu1 mu2];
noisy_cluster_mean = true_cluster_mean + randn(size(true_cluster_mean)) * noise_std;
noisy_beta = beta_star + randn(size(beta_star)) * noise_std;
lambda_multiplier = 1;
dummy_label = zeros(n,1)+1;
% Run CHIME
[~, ~, ~, ~, ~, ~, ~, cluster_est_chime] = CHIME(data, data, dummy_label, cluster_1_ratio, noisy_cluster_mean, noisy_beta,  1, 0.1,100);
% Evaluate clustering accuracy
acc = get_bicluster_accuracy(cluster_est_chime, label_true)
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
