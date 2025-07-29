function [pdMatrix, iterations] = findPDMatrix(initialMatrix, delta)
% FINDPDMATRIX Iteratively modifies a matrix until its symmetric part is Positive Definite.
%
%   [pdMatrix, iterations] = findPDMatrix(initialMatrix, delta)
%
%   Inputs:
%     initialMatrix - The starting matrix (can be non-symmetric).
%     delta         - The value to add to 5% of the selected zero entries in each iteration.
%
%   Outputs:
%     pdMatrix      - The resulting positive definite matrix (or its symmetric part).
%     iterations    - The number of iterations taken to find a PD matrix.
%
%   The function works by:
%   1. Identifying all zero entries in the lower triangular part of the current matrix.
%   2. Randomly selecting 5% of these zero entries.
%   3. Adding 'delta' to the selected zero entries.
%   4. Checking if the symmetric part of the modified matrix is positive definite.
%     (A matrix M is positive definite if all eigenvalues of (M+M')/2 are positive).
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
        rng(iterations); % Using 'iterations' as seed for reproducibility of the sequence

        % --- MODIFICATION START ---
        % Find indices of all zero entries in the LOWER TRIANGULAR part of the current matrix
        % tril(currentMatrix) creates a matrix with only the lower triangular part (and zeros elsewhere)
        % find(...) returns linear indices of non-zero elements.
        % We are interested in zeros, so we look for (tril(currentMatrix) == 0)
        % AND exclude the diagonal (which is part of tril but usually not considered "triangular" for this purpose)
        
        % Create a logical mask for the lower triangular part (excluding diagonal)
        lowerTriangularMask = tril(true(size(currentMatrix)), -1); % -1 excludes the main diagonal

        % Find zero entries specifically within this lower triangular part
        zeroIndicesInLowerTri = find(currentMatrix == 0 & lowerTriangularMask);

        % If there are no zero entries in the lower triangular part and it's not PD,
        % it might be an infinite loop or impossible to achieve PD this way.
        if isempty(zeroIndicesInLowerTri) && iterations > 1
            warning('No zero entries left in the lower triangular part to modify, and the matrix is not yet positive definite. Exiting loop.');
            break;
        end
        % --- MODIFICATION END ---

        % Calculate 5% of the zero entries to modify
        numZerosToModify = ceil(0.05 * length(zeroIndicesInLowerTri));

        % Ensure we don't try to modify more zeros than available
        if numZerosToModify > length(zeroIndicesInLowerTri)
            numZerosToModify = length(zeroIndicesInLowerTri);
        end

        % Randomly select indices of zero entries to modify
        if ~isempty(zeroIndicesInLowerTri) && numZerosToModify > 0
            selectedIndices = zeroIndicesInLowerTri(randperm(length(zeroIndicesInLowerTri), numZerosToModify));

            % Add delta to the selected zero entries
            currentMatrix(selectedIndices) = currentMatrix(selectedIndices) + delta;
        end

        % Check for positive definiteness
        % A matrix A is positive definite if its symmetric part (A+A')/2 has all positive eigenvalues.
        symmetricPart = (currentMatrix + currentMatrix') / 2;

        % Calculate eigenvalues of the symmetric part
        eigenvalues = eig(symmetricPart);

        % Check if all eigenvalues are positive
        % Using a small tolerance (e.g., 1e-9) for floating-point comparisons is good practice
        isPositiveDefinite = all(eigenvalues > 1e-9); % Check if all eigenvalues are greater than a small positive number

        fprintf('Iteration %d: Matrix is %s positive definite.\n', iterations, ...
                        ternary(isPositiveDefinite, 'now', 'not yet'));

        % Optional: Add a break condition for maximum iterations to prevent infinite loops
        if iterations > 10000 % Set a reasonable limit
            warning('Maximum iterations reached. Could not find a positive definite matrix.');
            break;
        end
    end

    pdMatrix = symmetricPart; % Return the final modified matrix
    fprintf('Found a Positive Definite matrix after %d iterations.\n', iterations);
end

 