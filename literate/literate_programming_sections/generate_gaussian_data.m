function [X, y, mu1, mu2, mahala_dist, Omega_star, beta_star] = generate_gaussian_data(n, p, sep, model, seed, cluster_1_ratio)
%% generate_gaussian_data
% @export
% 
% 
% 
% %GENERATE_GAUSSIAN_DATA Generate two-class Gaussian data with structured precision 
% matrix
% 
% % 
% 
% % INPUTS:
% 
% %   n               - total number of samples
% 
% %   p               - number of variables
% 
% %   model           - 'ER' (Erdős–Rényi) or 'AR' (autoregressive)
% 
% %   seed            - random seed
% 
% %   cluster_1_size  - proportion of samples in class 1 (e.g., 0.5)
% 
% %
% 
% % OUTPUTS:
% 
% %   X           - n x p data matrix
% 
% %   y           - n x 1 vector of class labels (1 or 2)
% 
% %   Omega_star  - p x p precision matrix
% 
% %   beta_star   - sparse discriminant vector
% 
% 
% 
% 
    
    s = 10;  % number of nonzero entries in beta
    n1 = round(n * cluster_1_ratio);
    n2 = n - n1;
    y = [ones(n1, 1); 2 * ones(n2, 1)];
    % Generate Omega_star
    switch model
        case 'ER'
            Omega_star = get_precision_ER(p);
        case 'chain45'
            Omega_star = get_precision_band(p, 2, 0.45);
        otherwise
            error('Model must be ''ER'' or ''AR''.');
    end
    % Set beta_star
    beta_star = zeros(p, 1);
    
    Sigma = inv(Omega_star);
    M=sep/2/ sqrt( sum( Sigma(1:s,1:s),"all") );
    beta_star(1:s) = M;
    % Set class means
    mu1 = Omega_star \ beta_star;
    mu2 = -mu1;
    % Mahalanobis distance
    mahala_dist_sq = (mu1 - mu2)' * Omega_star * (mu1 - mu2);
    mahala_dist = sqrt(mahala_dist_sq);
    fprintf('Mahalanobis distance between mu1 and mu2: %.4f\n', mahala_dist);
    % Generate noise once
    rng(seed);
    Z = mvnrnd(zeros(p, 1), Sigma, n);  % n x p noise
    % Create mean matrix
    mean_matrix = [repmat(mu1', n1, 1); repmat(mu2', n2, 1)];
    % Final data matrix
    X = Z + mean_matrix;
end
