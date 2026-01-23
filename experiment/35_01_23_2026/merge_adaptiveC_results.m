function merge_adaptiveC_results(results_dir, output_file)
%MERGE_ADAPTIVEC_RESULTS Merge individual adaptiveC_bandit results into a single MAT file
% and compute final posterior Beta means for each feature.
%
%   merge_adaptiveC_results(results_dir, output_file)
%
%   INPUTS:
%       results_dir  - Directory containing individual .mat files from bandit runs
%       output_file  - Path to save merged results (MAT file)
%
%   OUTPUTS:
%       merged_table          - Concatenated results_table from all files
%       final_beta_means_mat  - Struct containing alpha/beta final means for each rep
%
%   Notes:
%       - Assumes each .mat file contains variables:
%           - results_table
%           - bandit.alpha
%           - bandit.beta
%       - Saves merged results and posterior means in one MAT file.

if nargin < 2
    output_file = fullfile(results_dir, 'merged_results_adaptiveC.mat');
end

files = dir(fullfile(results_dir, '*.mat'));
if isempty(files)
    error('No .mat files found in %s', results_dir);
end

merged_table = table();
final_beta_means_mat = struct();

fprintf('Merging %d files from %s ...\n', length(files), results_dir);

for i = 1:length(files)
    file_path = fullfile(results_dir, files(i).name);
    data = load(file_path);
    
    % Merge results_table
    if isfield(data, 'results_table')
        merged_table = [merged_table; data.results_table]; %#ok<AGROW>
    else
        warning('File %s does not contain ''results_table'', skipping.', files(i).name);
    end
    
    % Compute and store final posterior Beta means
    if isfield(data, 'bandit')
        bandit_obj = data.bandit;
        if isfield(bandit_obj, 'alpha') && isfield(bandit_obj, 'beta')
            beta_mean = bandit_obj.alpha ./ (bandit_obj.alpha + bandit_obj.beta);
            final_beta_means_mat.(sprintf('rep_%d', i)) = beta_mean;
        else
            warning('File %s: bandit object missing alpha or beta.', files(i).name);
        end
    else
        warning('File %s: no bandit object found.', files(i).name);
    end
end

fprintf('Merged %d entries.\n', height(merged_table));
fprintf('Computed final Beta means for %d repetitions.\n', length(fieldnames(final_beta_means_mat)));

% Save merged table and posterior means
save(output_file, 'merged_table', 'final_beta_means_mat', '-v7.3');
fprintf('Saved merged results and posterior means to %s\n', output_file);

end
