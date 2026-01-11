function run_bandit_task(task_id, param_file, results_dir) % Add
    repository paths addpath(genpath('/home1/jongminm/sparse_kmeans'));
addpath(genpath('../../main_code'));

params = readmatrix(param_file);
C_val = params(task_id, 1);
p_val = params(task_id, 2);
rep = params(task_id, 3);

% Run simulation s = 10;
support = 1 : s;
[ data, label_true ] = generate_gaussian_data(
    200, p_val, s, 4, 'iso', 'equal_symmetric', 0, rep, 0.5, rep);

    bandit = sdp_kmeans_bandit(data', 2, C_val);
    bandit.fit_predict(20, label_true');
    
    % Save result to individual .mat file
    bandit.save_results(results_dir, rep, 4, support);
end
