function is_loop = detect_loop(obj_val_vec, loop_detect_start, window_size, min_delta)
%% detect_loop
% @export
% 
% Detects convergence plateau based on recent objective values.
% 
% *Inputs:*
%% 
% * |obj_val_vec|       - Vector of objective values over iterations (may contain 
% NaN)
% * |loop_detect_start| - Number of initial steps to skip before detecting loops
% * |window_size|       - Number of recent steps to consider
% * |min_delta|    - Minimum required relative improvement (in percent) to avoid 
% detection
%% 
% *Output:*
%% 
% * |is_loop|          - Logical flag: true if no significant improvement is 
% detected
%% 
% *Description:*
%% 
% * Resembles |keras.callbacks.EarlyStopping| (<https://keras.io/api/callbacks/early_stopping/ 
% https://keras.io/api/callbacks/early_stopping/>), incorporating the |min_delta| 
% parameter. In our setting, a window of iterations plays the role of epochs in 
% deep learning.
% * If the last |window_size| iterations show no improvement over the global 
% optimum so far, return the flag |is_loop|.
%% 
% 
    is_loop = false; % Default output
    % Trim input at first NaN, if any
    nan_idx = find(isnan(obj_val_vec), 1, 'first');
    if ~isempty(nan_idx)
        obj_val_vec = obj_val_vec(1:nan_idx - 1);
    end
    n = numel(obj_val_vec);
    % Check if there's enough history
    if n <= loop_detect_start + window_size
        return;
    end
    % Define the best value before the recent window
    global_best = max(obj_val_vec(1:end - window_size));
    % Define the best value in the recent window
    window_vec = obj_val_vec(end - window_size + 1:end);
    window_best = max(window_vec);
    % Compute relative change
    relative_change = abs(global_best - window_best) / max(abs(global_best), eps);
    % Determine if loop (stagnation) is happening
    if relative_change < min_delta
        is_loop = true;
    end
end
% 
%% 
% 
