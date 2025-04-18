function test_get_precision_ER()
%% test_get_precision_ER
% @export
%TEST_GET_PRECISION_ER Tests the get_precision_ER function for correctness
    p = 100;  % Dimensionality to test
    Omega = get_precision_ER(p);
    % Check symmetry
    if ~isequal(Omega, Omega')
        error('Test failed: Omega is not symmetric.');
    else
        disp('✓ Symmetry test passed.');
    end
    % Check positive definiteness
    eigenvalues = eig(Omega);
    if all(eigenvalues > 0)
        disp('✓ Positive definiteness test passed.');
    else
        error('Test failed: Omega is not positive definite.');
    end
    % Check unit diagonal
    diag_vals = diag(Omega);
    if max(abs(diag_vals - 1)) < 1e-10
        disp('✓ Unit diagonal test passed.');
    else
        error('Test failed: Omega is not standardized to unit diagonal.');
    end
    % Check approximate sparsity level (off-diagonal non-zeros)
    is_offdiag = ~eye(p);
    num_nonzeros_offdiag = nnz(Omega .* is_offdiag);
    expected_nonzeros = round(0.01 * p^2);
    
    if abs(num_nonzeros_offdiag - expected_nonzeros) / expected_nonzeros < 0.2
        disp(['✓ Sparsity test passed: ' num2str(num_nonzeros_offdiag) ...
              ' non-zero off-diagonal entries (expected ~' num2str(expected_nonzeros) ').']);
    else
        error(['Test failed: Unexpected number of off-diagonal non-zeros (' ...
               num2str(num_nonzeros_offdiag) ' vs expected ' num2str(expected_nonzeros) ').']);
    end
    % Optional: visualize sparsity pattern
    figure;
    spy(Omega);
    title('Sparsity Pattern of Precision Matrix \Omega^*');
    xlabel('Column');
    ylabel('Row');
end
%% 
% 
%% 
% 
% 
% 
% 
% 
