function relative_change = get_relative_change(obj_val_vec)
%% get_relative_change
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
% |relative_change = get_relative_change(obj_val_vec)|
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
    % Find index of the first NaN
    nan_idx = find(isnan(obj_val_vec), 1, 'first');
    if isempty(nan_idx)
        valid_vals = obj_val_vec;
    else
        valid_vals = obj_val_vec(1:nan_idx - 1);
    end
    if numel(valid_vals) < 2
        relative_change = Inf;
        warning('get_relative_change:InsufficientValidLength', ...
                'Fewer than two valid values before NaN. Returning Inf.');
        return;
    end
    prev_val = valid_vals(end - 1);
    curr_val = valid_vals(end);
    % Use max with eps to ensure numerical stability
    relative_change = abs(curr_val - prev_val) / max(abs(prev_val), eps);
end
%% 
% 
%% 
% 
%% 
