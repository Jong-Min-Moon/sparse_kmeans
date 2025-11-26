%% ISEE_kmeans_clean_simul_rand
% @export
function cluster_estimate = ISEE_kmeans_clean_simul_rand(x, k, n_iter, is_parallel, loop_detect_start, window_size, min_delta, db_dir, table_name, rep, model, sep, cluster_true)
% ISEE_kmeans_clean - Runs iterative clustering with early stopping and logs results to SQLite DB
    [p, n] = size(x);  % Get dimensions
    obj_sdp = nan(1, n_iter);
    obj_lik = nan(1, n_iter);
    % Initialize cluster assignment
    cluster_estimate_now = sdp_kmeans(x, k);
    is_stop = 0;
       iter=0;
            rand_vec = nan(1, n_iter);
    while (~is_stop) && (iter < n_iter)
        iter= iter+1;
        [cluster_estimate, s_hat, obj_sdp(iter), obj_lik(iter)] = ISEE_kmeans_clean_onestep(x, k, cluster_estimate_now, is_parallel);
       %%%%%%%%%%%%%%%% simul part starts
        TP = sum(s_hat(1:10));
        FP = sum(s_hat) - TP;
        FN = 10 - TP;
acc = get_bicluster_accuracy(cluster_estimate, cluster_true);  % define this if needed
fprintf('Iteration %d | SDP obj: %.4f | Likelihood obj: %.4f | TP: %d | FP: %d | FN: %d | Acc: %.4f\n', ...
    iter, obj_sdp(iter), obj_lik(iter), TP, FP, FN, acc);
    
    % === Insert into SQLite database ===
    jobdate = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss');
    max_attempts = 10;
    pause_time = 5;
    attempt = 1;
    
    while attempt <= max_attempts
        try
            conn = sqlite(db_dir, 'connect');
            insert_query = sprintf([ ...
                'INSERT INTO %s (rep, iter, sep, dim, n, model, acc, obj_sdp, obj_lik, true_pos, false_pos, false_neg, jobdate) ' ...
                'VALUES (%d, %d, %d, %d, %d, "%s", %.6f, %.6f, %.6f, %d, %d, %d, "%s")'], ...
                table_name, rep, iter, sep, p, n, model, acc, obj_sdp(iter), obj_lik(iter), TP, FP, FN, char(jobdate));
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
    %%%%%%%%%%%%% simul part ends
       % Early stopping condition
       rand_score = RandIndex(cluster_estimate_now, cluster_estimate)
       rand_vec(iternum) = rand_score;
        is_stop = decide_stop_rand(rand_vec, loop_detect_start, window_size, min_delta);
        cluster_estimate = cluster_estimate_now;
        if is_stop
            break;
        end
    end
end
%% 
