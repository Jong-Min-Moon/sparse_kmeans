function [discov_true_vec, discov_false_vec] = get_discovery(entries_survived, s)
    n_iter = size(entries_survived, 1);
    discov_true_vec = zeros(n_iter, 1);
    discov_false_vec = zeros(n_iter, 1);
    for i = 1:n_iter
        entries_survived_now = entries_survived(i,:);
        discov_true_vec(i) = sum(entries_survived_now(1:s));
        discov_false_vec(i) = sum(entries_survived_now) - discov_true_vec(i);
    end