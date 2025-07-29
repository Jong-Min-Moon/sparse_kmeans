function [pdMatrix, iterations] = findPDMatrix(initialMatrix, delta)
% FINDPDMATRIX Iteratively modifies a matrix until its symmetric part is Positive Definite.
%
%   [pdMatrix, iterations] = findPDMatrix(initialMatrix, delta)
%
%   Inputs:
%     initialMatrix - The starting matrix (can be non-symmetric).
%     delta         - The value to add to 10% of the zero entries in each iteration.
%
%   Outputs:
%     pdMatrix      - The resulting positive definite matrix (or its symmetric part).
%     iterations    - The number of iterations taken to find a PD matrix.
%
%   The function works by:
%   1. Identifying all zero entries in the current matrix.
%   2. Randomly selecting 10% of these zero entries.
%   3. Adding 'delta' to the selected zero entries.
%   4. Checking if the symmetric part of the modified matrix is positive definite.
%      (A matrix M is positive definite if all eigenvalues of (M+M')/2 are positive).
%   5. Repeating the process with a new random seed until a PD matrix is found.

    % Validate inputs
    if ~ismatrix(initialMatrix)
        error('Input initialMatrix must be a matrix.');
    end
    if ~isscalar(delta) || ~isnumeric(delta)
        error('Input delta must be a numeric scalar.');
    end

    currentMatrix = initialMatrix;
    iterations = 0;
    isPositiveDefinite = false;

    fprintf('Starting search for a Positive Definite matrix...\n');

    % Loop until a positive definite matrix is found
    while ~isPositiveDefinite
        iterations = iterations + 1;

        % Set a new random seed for each iteration to ensure different zero selections
        % 'shuffle' uses the current time to seed the generator, making it different each time.
        rng(iterations);

        % Find indices of all zero entries in the current matrix
        % 'find' returns linear indices. If you need row, col: [row, col] = find(currentMatrix == 0);
        zeroIndices = find(currentMatrix == 0);

        % If there are no zero entries left and it's not PD, it might be an infinite loop
        if isempty(zeroIndices) && iterations > 1
            warning('No zero entries left to modify, and the matrix is not yet positive definite. Exiting loop.');
            break;
        end

        % Calculate 10% of the zero entries to modify
        numZerosToModify = ceil(0.10 * length(zeroIndices));

        % Ensure we don't try to modify more zeros than available
        if numZerosToModify > length(zeroIndices)
            numZerosToModify = length(zeroIndices);
        end

        % Randomly select indices of zero entries to modify
        % randperm generates a random permutation of integers
        if ~isempty(zeroIndices)
            selectedIndices = zeroIndices(randperm(length(zeroIndices), numZerosToModify));

            % Add delta to the selected zero entries
            currentMatrix(selectedIndices) = currentMatrix(selectedIndices) + delta;
        end

        % Check for positive definiteness
        % A matrix A is positive definite if its symmetric part (A+A')/2 has all positive eigenvalues.
        symmetricPart = (currentMatrix + currentMatrix') / 2;

        % Calculate eigenvalues of the symmetric part
        eigenvalues = eig(symmetricPart);

        % Check if all eigenvalues are positive
        % Using a small tolerance (e.g., eps) for floating-point comparisons is good practice
        isPositiveDefinite = all(eigenvalues > 1e-9); % Check if all eigenvalues are greater than a small positive number

        fprintf('Iteration %d: Matrix is %s positive definite.\n', iterations, ...
                ternary(isPositiveDefinite, 'now', 'not yet'));

        % Optional: Add a break condition for maximum iterations to prevent infinite loops
        if iterations > 10000 % Set a reasonable limit
            warning('Maximum iterations reached. Could not find a positive definite matrix.');
            break;
        end
    end

    pdMatrix = currentMatrix;
    fprintf('Found a Positive Definite matrix after %d iterations.\n', iterations);
end

% Helper function for ternary operation (not strictly necessary but can make code cleaner)