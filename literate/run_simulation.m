function run_simulation()
    
    % --- Simulation Parameters ---
    n = 200;   % Samples
    p = 4000;  % Dimensions
    s = 10;    % True Sparsity
    K = 2;     % Clusters
    
    target_l2 = 4.0;
    sep = target_l2 / sqrt(s); % Per-dimension separation
    n_trials = 10;
    
    fprintf('Running B&B Simulation (%d Trials)...\n', n_trials);
    fprintf('n=%d, p=%d, s=%d, K=%d, Target L2=%.1f\n', n, p, s, K, target_l2);
    
    % Storage
    results_iter_fixed  = struct('acc', [], 'tp', [], 'fp', [], 'obj', []);
    results_bnb_s       = struct('acc', [], 'tp', [], 'fp', [], 'obj', [], 'time', []);
    results_bnb_nos     = struct('acc', [], 'tp', [], 'fp', [], 'obj', [], 'time', [], 's_est', []);
    
    for trial = 1:n_trials
        rng(trial);
        fprintf('\n=== Trial %d/%d ---\n', trial, n_trials);
        
        % --- Generate Data ---
        X = randn(p, n);
        true_features = 1:s;
        X(true_features, 1:n/2) = X(true_features, 1:n/2) + sep/2;
        X(true_features, n/2+1:n) = X(true_features, n/2+1:n) - sep/2;
        
        mu1 = mean(X(true_features, 1:n/2), 2);
        mu2 = mean(X(true_features, n/2+1:n), 2);
        emp_l2 = norm(mu1 - mu2);
        fprintf('Empirical L2 Separation: %.4f\n', emp_l2);
        
        true_labels = [ones(1, n/2), 2*ones(1, n/2)];

        % --- 1. Baseline: Iterative SDP (Fixed Sparsity) ---
        fprintf('\n[1/3] Running Iterative SDP (Fixed s=%d)...\n', s);
        iter_fixed = sdp_kmeans_iter_fixedsparsity(X, K, s);
        [~, labels_f] = iter_fixed.fit_predict(10, 5, 3, 1e-3);
        
        feat_f = iter_fixed.selected_features;
        acc_f = sum(labels_f(:)' == true_labels) / n;
        if acc_f < 0.5, acc_f = 1 - acc_f; end
        tp_f = length(intersect(feat_f, true_features));
        X_f = X(feat_f, :); Z_f = kmeans_sdp_pengwei(X_f'*X_f, K);
        obj_f = sum(sum((X_f*Z_f).*X_f));
        
        results_iter_fixed.acc(end+1) = acc_f;
        results_iter_fixed.tp(end+1) = tp_f;
        results_iter_fixed.obj(end+1) = obj_f;

        % --- 2. Proposed: B&B (Known s) ---
        fprintf('\n[2/3] Running B&B (Known s=%d)...\n', s);
        % s-constraint mode (use_penalty = false)
        bnb_s = sdp_kmeans_bnb(X, K, s, false);
        tic;
        [labels_bs, feat_bs, log_bs, init_bs] = bnb_s.solve();
        results_bnb_s.time(end+1) = toc;
        
        acc_bs = sum(labels_bs(:)' == true_labels) / n;
        if acc_bs < 0.5, acc_bs = 1 - acc_bs; end
        results_bnb_s.acc(end+1) = acc_bs;
        results_bnb_s.tp(end+1) = length(intersect(feat_bs, true_features));
        results_bnb_s.obj(end+1) = log_bs.obj;

        % --- 3. Proposed: B&B (Unknown s, Penalty) ---
        fprintf('\n[3/3] Running B&B (Unknown s, Auto Penalty)...\n');
        % penalty mode (use_penalty = true, lambda = [] to auto-calc)
        bnb_nos = sdp_kmeans_bnb(X, K, [], true);
        tic;
        [labels_bn, feat_bn, log_bn, init_bn] = bnb_nos.solve();
        results_bnb_nos.time(end+1) = toc;
        
        acc_bn = sum(labels_bn(:)' == true_labels) / n;
        if acc_bn < 0.5, acc_bn = 1 - acc_bn; end
        results_bnb_nos.acc(end+1) = acc_bn;
        results_bnb_nos.tp(end+1) = length(intersect(feat_bn, true_features));
        results_bnb_nos.obj(end+1) = log_bn.obj;
        results_bnb_nos.s_est(end+1) = length(feat_bn);

        fprintf('\nTrial %d Summary:\n', trial);
        fprintf('  Iter(Fixed): Obj=%.2f, Acc=%.2f%%, TP=%d\n', obj_f, acc_f*100, tp_f);
        fprintf('  B&B (s):     Obj=%.2f (Init: %.2f), Acc=%.2f%%, TP=%d\n', log_bs.obj, init_bs.obj, acc_bs*100, results_bnb_s.tp(end));
        fprintf('  B&B (No-s):  Obj=%.2f (Init: %.2f), Acc=%.2f%%, TP=%d, s_est=%d\n', log_bn.obj, init_bn.obj, acc_bn*100, results_bnb_nos.tp(end), results_bnb_nos.s_est(end));
    end
    
    fprintf('\n' + repmat('=', 1, 60) + '\n');
    fprintf('FINAL COMPARISON (Averages over %d trials)\n', n_trials);
    fprintf('%-15s | %-12s | %-12s | %-12s\n', 'Metric', 'Iter(Fixed)', 'B&B (s)', 'B&B (No-s)');
    fprintf('%-15s | %-12s | %-12s | %-12s\n', '-'*15, '-'*12, '-'*12, '-'*12);
    fprintf('%-15s | %12.2f | %12.2f | %12.2f\n', 'Avg Objective', mean(results_iter_fixed.obj), mean(results_bnb_s.obj), mean(results_bnb_nos.obj));
    fprintf('%-15s | %11.2f%% | %11.2f%% | %11.2f%%\n', 'Avg Accuracy', mean(results_iter_fixed.acc)*100, mean(results_bnb_s.acc)*100, mean(results_bnb_nos.acc)*100);
    fprintf('%-15s | %12.2f | %12.2f | %12.2f\n', 'Avg True Pos', mean(results_iter_fixed.tp), mean(results_bnb_s.tp), mean(results_bnb_nos.tp));
    fprintf('%-15s | %12s | %12s | %12.2f\n', 'Avg s_est', s, s, mean(results_bnb_nos.s_est));
    fprintf('%-15s | %12s | %12.2f | %12.2f\n', 'Avg Time (s)', 'N/A', mean(results_bnb_s.time), mean(results_bnb_nos.time));
    fprintf(repmat('=', 1, 60) + '\n');
end
