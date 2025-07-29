% Main Script for Measuring Code Timing

% --- Configuration ---
n_rep = 100; % Number of repetitions for each 'n' value
p = 5000;    % Number of features (fixed)
n_values = [100 500 1000 2000]; % Different sample sizes to test
n_iter = 20; % Fixed number of iterations for the clustering algorithm

% Initialize a table or struct to store the results
timing_results = table(...
    zeros(numel(n_values), 1), ... % Column for 'n'
    zeros(numel(n_values), 1), ... % Column for 'AverageTimePerRep'
    'VariableNames', {'SampleSize_n', 'AverageTimePerRep_sec'});

disp('Starting timing measurements...');
disp('----------------------------------------------------');

% Outer loop: Iterate through different sample sizes (n)
for idx_n = 1:numel(n_values)
    n = n_values(idx_n); % Current sample size

    fprintf('Processing n = %d (p = %d). Running %d repetitions...\n', n, p, n_rep);

    % Array to store times for each repetition
    times_for_current_n = zeros(1, n_rep);

    % Inner loop: Perform 'n_rep' repetitions for the current 'n'
    for i = 1:n_rep
        % Start timer for the current repetition
        tic;

        % --- Data Generation ---
        % The 4, 1, 0.5 are dummy parameters for the generator,
        % 1 and 0 for get_data (seed and varargin)
        generator = data_generator_approximately_sparse_mean(n, p, 10, 4, i, 0.5);
        [data, label_true] = generator.get_data(1, 0); % Use 'i' as seed for unique data per rep

        % --- Clustering ---
        clusterer = sdp_kmeans_iter_knowncov(data, 2); % 2 is k_clusters
        initial_clustering = kmeans(data', 2);
        % ${T} is replaced by n_iter as per the pseudocode.
        % This will run the fit_predict method for n_iter times.
        cluster_est = clusterer.fit_predict(n_iter, initial_clustering');

        % Stop timer for the current repetition
        elapsed_time_rep = toc;
        
        times_for_current_n(i) = elapsed_time_rep;

        % Optional: Display progress for long runs
        if mod(i, 10) == 0 || i == n_rep
            fprintf('  Repetition %d/%d: %.4f seconds\n', i, n_rep, elapsed_time_rep);
        end
    end

    % Calculate the average time for the current 'n'
    average_time_n = mean(times_for_current_n);
    std_dev_time_n = std(times_for_current_n); % Optional: also calculate std dev

    fprintf('Average time for n = %d: %.4f seconds (Std Dev: %.4f)\n', n, average_time_n, std_dev_time_n);
    disp('----------------------------------------------------');

    % Store results
    timing_results.SampleSize_n(idx_n) = n;
    timing_results.AverageTimePerRep_sec(idx_n) = average_time_n;
end

disp('Timing measurements complete. Summary:');
disp(timing_results);

% Clean up (optional)
clear n_rep p n_values n_iter generator clusterer data label_true cluster_est;