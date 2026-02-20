function test_nos_bnb()
    % --- Verification Parameters ---
    n = 200;   
    p = 1000;  
    s = 10;    
    K = 2;     
    target_l2 = 6.0;
    sep = target_l2 / sqrt(s);
    
    fprintf('=== Verification: No-S B&B Penalty Mode ===\n');
    fprintf('n=%d, p=%d, true s=%d, target L2=%.1f\n', n, p, s, target_l2);
    
    % --- Generate Data ---
    rng(42); % For reproducibility
    X = randn(p, n);
    true_features = 1:s;
    X(true_features, 1:n/2) = X(true_features, 1:n/2) + sep/2;
    X(true_features, n/2+1:n) = X(true_features, n/2+1:n) - sep/2;
    
    true_labels = [ones(1, n/2), 2*ones(1, n/2)];
    
    % --- Initialize Solver ---
    % use_penalty = true, s_or_lambda = [] (auto-determine)
    bnb_nos = sdp_kmeans_bnb(X, K, [], true);
    
    % --- Solve ---
    tic;
    [labels, feat, log, initial] = bnb_nos.solve();
    total_time = toc;
    
    % --- Metrics ---
    acc = sum(labels(:)' == true_labels) / n;
    if acc < 0.5, acc = 1 - acc; end
    
    tp = length(intersect(feat, true_features));
    fp = length(feat) - tp;
    s_est = length(feat);
    
    fprintf('\nResults:\n');
    fprintf('  Initial Obj: %.4f (Heuristic)\n', initial.obj);
    fprintf('  Final Obj:   %.4f (B&B)\n', log.obj);
    fprintf('  Accuracy:    %.2f%%\n', acc * 100);
    fprintf('  True s:      %d\n', s);
    fprintf('  Est s:       %d (TP=%d, FP=%d)\n', s_est, tp, fp);
    fprintf('  Nodes:       %d\n', log.nodes);
    fprintf('  Time:        %.2f s\n', total_time);
    
    if acc > 0.9 && tp >= 8 && log.obj >= initial.obj
        fprintf('\nVERIFICATION PASSED!\n');
    else
        fprintf('\nVERIFICATION FAILED or Performance sub-optimal.\n');
    end
end
