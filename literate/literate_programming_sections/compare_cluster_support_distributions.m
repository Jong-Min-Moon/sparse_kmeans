function compare_cluster_support_distributions(n, p, s, sep, baseline, cluster_1_ratio, true_support, false_support, n_rep, beta_seed)
%% compare_cluster_support_distributions
% @export
% Compare objective value distributions for likelihood and SDP
% under 20%, 40%, and 50% label flips. Plot 8 histograms (2x4 layout).
flip_rates = [0.2, 0.4, 0.5, 1];
% Generate a general random cluster estimate (1 or 2)
cluster_rand = randi([1, 2], n, 1);
[X_full, y_true, ~, ~, ~, ~, ~] = generate_gaussian_data(n, p, s, sep, 'iso', 'random_uniform', baseline, 1, cluster_1_ratio, 1);
% Allocate results structure
results_lik = struct();
results_sdp = struct();
for f = 1:length(flip_rates)
    flip_rate = flip_rates(f);
    % Allocate
    obj_lik_tt = zeros(n_rep, 1);
    obj_lik_tr = zeros(n_rep, 1);
    obj_lik_ft = zeros(n_rep, 1);
    obj_lik_fr = zeros(n_rep, 1);
    
    obj_sdp_tt = zeros(n_rep, 1);
    obj_sdp_tr = zeros(n_rep, 1);
    obj_sdp_ft = zeros(n_rep, 1);
    obj_sdp_fr = zeros(n_rep, 1);
    
    % Flip cluster labels
        y_flip = y_true;
        n_flip = round(flip_rate * n);
        flip_idx = randperm(n, n_flip);
        y_flip(flip_idx) = 3 - y_flip(flip_idx);  % flip 1 <-> 2
        if flip_rate ==1
            y_flip = cluster_rand;
        end
    for seed = 1:n_rep
        [X_full, y_true, ~, ~, ~, ~, ~] = generate_gaussian_data(n, p, s, sep, 'iso', 'random_uniform', seed, cluster_1_ratio, beta_seed);
        X_true = X_full(:, true_support);
        X_false = X_full(:, false_support);
        % Likelihood
        obj_lik_tt(seed) = get_likelihood_objective(X_true', y_true);
        obj_lik_tr(seed) = get_likelihood_objective(X_true', y_flip);
        obj_lik_ft(seed) = get_likelihood_objective(X_false', y_true);
        obj_lik_fr(seed) = get_likelihood_objective(X_false', y_flip);
        % SDP
        obj_sdp_tt(seed) = get_sdp_objective(X_true', y_true);
        obj_sdp_tr(seed) = get_sdp_objective(X_true', y_flip);
        obj_sdp_ft(seed) = get_sdp_objective(X_false', y_true);
        obj_sdp_fr(seed) = get_sdp_objective(X_false', y_flip);
     
    end
    % Store each mode separately
    results_lik(f).flip_rate = flip_rate;
    results_lik(f).vals = {obj_lik_tt, obj_lik_tr, obj_lik_ft, obj_lik_fr};
    results_sdp(f).flip_rate = flip_rate;
    results_sdp(f).vals = {obj_sdp_tt, obj_sdp_tr, obj_sdp_ft, obj_sdp_fr};
end
fig = figure('Position', [100, 100, 1800, 900]);
t = tiledlayout(fig, 2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
labels = {'True+True', 'True+Flip', 'False+True', 'False+Flip'};
hist_handles = gobjects(1, 4);  % Store handles for common legend
for i = 1:length(flip_rates)
    % --- Likelihood Subplot ---
    nexttile(t, i);
    hold on;
    vals = results_lik(i).vals;
    for j = 1:4
        h = histogram(vals{j}, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
        if i == 1  % store histogram handles from the first subplot
            hist_handles(j) = h;
        end
    end
    if flip_rates(i) == 1
        title('Likelihood | Random Guess');
    else
        title(sprintf('Likelihood | %.0f%% Flip', 100 * flip_rates(i)));
    end
    xlabel('Likelihood Objective');
    ylabel('Density');
    grid on;
    set(gca, 'FontSize', 14);
    % --- SDP Subplot ---
    nexttile(t, i + 4);
    hold on;
    vals = results_sdp(i).vals;
    for j = 1:4
        histogram(vals{j}, 'Normalization', 'pdf', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    end
    if flip_rates(i) == 1
        title('SDP | Random Guess');
    else
        title(sprintf('SDP | %.0f%% Flip', 100 * flip_rates(i)));
    end
    xlabel('SDP Objective');
    ylabel('Density');
    grid on;
    set(gca, 'FontSize', 14);
end
% Add external legend below all plots
lgd = legend(hist_handles, labels, 'Orientation', 'horizontal', 'FontSize', 16);
lgd.Layout.Tile = 'south';
% Remove axes toolbar interactivity
ax_list = findall(fig, 'Type', 'axes');
for ax = ax_list'
    disableDefaultInteractivity(ax);
end
% --- Verify figure validity before saving ---
if ~isvalid(fig)
    error('Figure handle is invalid before exporting.');
end
% --- Save figure ---
fname = sprintf('objective_dists_s%d_ratio%.2f_false%d_%d.png', ...
    s, cluster_1_ratio, false_support(1), false_support(end));
fname = strrep(fname, ' ', '');
fname = strrep(fname, '[', '');
fname = strrep(fname, ']', '');
exportgraphics(fig, fname, 'Resolution', 300);
fprintf('Saved figure to: %s\n', fname);
end
%% 
% 
% 
% 
