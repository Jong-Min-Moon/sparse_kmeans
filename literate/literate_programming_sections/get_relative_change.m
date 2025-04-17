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
% 
% This function is typically used in optimization algorithms to monitor convergence. 
% It calculates the relative difference between the two most recent objective 
% values. A small relative change suggests that the algorithm is approaching convergence.
% 
% 
% 
% 
    if numel(obj_val_vec) < 2
        relative_change = Inf;
        warning('get_relative_change:InsufficientLength', ...
                'Input vector must contain at least two elements. Returning Inf.');
        return;
    end
    prev_val = obj_val_vec(end - 1);
    curr_val = obj_val_vec(end);
    
    % Use max with eps to ensure numerical stability
    relative_change = abs(curr_val - prev_val) / max(abs(prev_val), eps);
end
%% 
% 
%% 
% 
