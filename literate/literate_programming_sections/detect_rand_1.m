%% detect_rand_1
% @export
% 
% 
function is_stuck = detect_rand_1(obj_val_vec, detect_start, window_size)
% detect_relative_change - Checks whether the last two valid objective values
% show insufficient relative improvement.
%
% Inputs:
%   obj_val_vec - Vector of objective values (may contain NaNs)
%   min_delta   - Minimum required relative change to count as progress
%
% Output:
%   is_stuck    - Logical flag: true if relative change < min_delta
    % Trim at first NaN, if any
    nan_idx = find(isnan(obj_val_vec), 1, 'first');
    if isempty(nan_idx)
        valid_vals = obj_val_vec;
    else
        valid_vals = obj_val_vec(1:nan_idx - 1);
    end
    % Need at least two valid values to compute relative change
    if numel(valid_vals) < max(2, detect_start)
        is_stuck = false;
        return;
    end
    % Compute relative change
  
    curr_val = valid_vals(end-window_size : end);
    relative_change = abs(1-mean(curr_val));
    % Determine if change is below threshold
    is_stuck = relative_change < 0.01;
end
