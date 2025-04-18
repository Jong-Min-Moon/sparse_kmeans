function precision_matrix = get_precision_band(p, precision_sparsity, conditional_correlation)
%% get_precision_band
% @export
% 
% 
% get_precision_band - Constructs a banded symmetric precision matrix with geometric decay
%                      using spdiags, assuming identity base and symmetric off-diagonal decay.
%
% Inputs:
%   p                    - Dimension of the matrix
%   precision_sparsity   - Total number of nonzero diagonals (must be even, e.g., 2, 4, 6, ...)
%   conditional_correlation - Decay factor for off-diagonals
%
% Output:
%   precision_matrix     - p × p symmetric precision matrix
    if precision_sparsity < 2
        precision_matrix = eye(p);
        return;
    end
    max_band = floor(precision_sparsity / 2);       % e.g., 2 → 1 band above/below
    offsets = -max_band:max_band;                   % Diagonal offsets
    num_diags = length(offsets);
    % Create padded diagonal matrix: size p × num_diags
    B = zeros(p, num_diags);
    for k = 1:num_diags
        offset = offsets(k);
        len = p - abs(offset);
        decay = conditional_correlation ^ abs(offset);
        B((1:len) + max(0, offset), k) = decay;
    end
    % Build matrix from diagonals
    precision_matrix = full(spdiags(B, offsets, p, p));
end
