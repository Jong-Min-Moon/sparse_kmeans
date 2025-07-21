function test_generate_gaussian_data()
%% test_generate_gaussian_data
% @export
% 
% 
%TEST_GENERATE_GAUSSIAN_DATA Test data generation for different p values
    n = 200;
    ps = [100, 200, 500, 800];
    seed = 42;
    cluster_1_ratio = 0.5;
    model = 'ER';
    for i = 1:length(ps)
        p = ps(i);
        fprintf('\n--- Testing with p = %d ---\n', p);
        [X, y, Omega_star, beta_star] = generate_gaussian_data(n, p, model, seed, cluster_1_ratio);
        % Check dimensions
        assert(isequal(size(X), [n, p]), 'Data matrix X has incorrect dimensions.');
        assert(isequal(size(y), [n, 1]), 'Label vector y has incorrect dimensions.');
        assert(isequal(size(Omega_star), [p, p]), 'Precision matrix has incorrect dimensions.');
        assert(isequal(size(beta_star), [p, 1]), 'Discriminant vector beta_star has incorrect dimensions.');
        % Basic checks
        fprintf('Number of samples in class 1: %d\n', sum(y == 1));
        fprintf('Number of samples in class 2: %d\n', sum(y == 2));
        fprintf('Number of non-zero entries in beta_star: %d\n', nnz(beta_star));
        % Optional: check symmetry and positive definiteness
        if ~isequal(Omega_star, Omega_star')
            warning('Omega_star is not symmetric.');
        end
        if any(eig(Omega_star) <= 0)
            warning('Omega_star is not positive definite.');
        end
    end
end
%% 
