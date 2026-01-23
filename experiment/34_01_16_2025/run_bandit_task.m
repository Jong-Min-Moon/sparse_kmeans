function run_bandit_task(task_id, param_file, results_dir) 
    addpath(genpath('/home1/jongminm/sparse_kmeans'));
    
    % --- DEBUGGING LINES ---
    check_file = which('sdp_kmeans_bandit');
    if isempty(check_file)
        error('MATLAB cannot find sdp_kmeans_bandit.m. Check filename and path.');
    else
        fprintf('Found class at: %s\n', check_file);
    end
    % -----

params = readmatrix(param_file);
C_val = params(task_id, 1);
p_val = params(task_id, 2);
rep = params(task_id, 3);

% Run simulation 
s = 10;
support = 1:s;
[ data, label_true ] = generate_gaussian_data(200, p_val, s, 4, 'iso', 'equal_symmetric', 0, rep, 0.5, rep);

    bandit = sdp_kmeans_bandit(data', 2, C_val);
    bandit.fit_predict(1000, label_true');
    
    % Save result to individual .mat file
    bandit.save_results(results_dir, rep, 4, support);
end
