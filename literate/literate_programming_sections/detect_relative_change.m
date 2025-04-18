function is_stuck = detect_relative_change(obj_val_vec, detect_start, min_delta)
%% detect_relative_change
% @export
% 
% 
% 
% Computes the relative change in the objective value between the last two iterations.
% 
% 
% 
% *Syntax:*
% 
% |is_stuck = get_relative_change(obj_val_vec)|
% 
% 
% 
% *Input:*
%% 
% * |obj_val_vec| - Numeric vector of objective values over iterations (length 
% must be >= 2).
%% 
% *Output:*
%% 
% * |relative_change| - The relative change between the last two objective values
%% 
% *Description:*
%% 
% * This function is typically used in optimization algorithms to monitor convergence. 
% It calculates the relative difference between the two most recent objective 
% values. A small relative change suggests that the algorithm is approaching convergence. 
% * If the vector contains NaNs, the computation is based on the last two values 
% before the first NaN.
% * If fewer than two valid values exist, returns Inf.
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
    prev_val = valid_vals(end - 1);
    curr_val = valid_vals(end);
    relative_change = abs(curr_val - prev_val) / max(abs(prev_val), eps);
    % Determine if change is below threshold
    is_stuck = relative_change < min_delta;
end
%% 
% 
%% 
% 
%% 
