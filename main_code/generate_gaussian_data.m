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
% *Inputs:*
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
% *Outputs:*
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
function [X, y, mu1, mu2, mahala_dist, Omega_star, beta_star] = generate_gaussian_data(n, p, s, sep, model_cov, model_energy, baseline, seed, cluster_1_ratio, beta_seed)
  
    n1 = round(n * cluster_1_ratio);
    n2 = n - n1;
    y = [ones(n1, 1); 2 * ones(n2, 1)];
    % Generate Omega_star
    switch model_cov
        case 'iso'
            Omega_star = eye(p);
            Sigma = eye(p);
        case 'ER'
            Omega_star = get_precision_ER(p);
            Sigma = inv(Omega_star);
        case 'chain45'
            Omega_star = get_precision_band(p, 2, 0.45);
            Sigma = inv(Omega_star);
        case 'chain45_approx'
            Omega_star = get_precision_band(p, 2, 0.45);
            diag_Omega_star = diag(Omega_star);
            Omega_star= Omega_star + delta
            Omega_star(logical(eye(size(Omega_star)))) = diag_Omega_star;
            Sigma = inv(Omega_star);
        otherwise
            error('Model must be ''ER'' or ''AR''.');
    end
    % Set beta_star
    switch model_energy
        case 'equal_symmetric'
                beta_star = zeros(p, 1);
                beta_star(1:s) = 1;
                M=sep/2/ sqrt( sum( Sigma(1:s,1:s),"all") );
                beta_star = M * beta_star;
                    % Set class means
                mu1 = Omega_star \ beta_star;
                mu2 = -mu1;
    %        signal_beta_1 = zeros(10) + 0.5;
    %        signal_beta_2 = -signal_beta_1;
        case 'random_uniform'
            rng(beta_seed)
            signal_beta_1 = 10 * rand(1, s) - 5;
            signal_beta_2 = 10 * rand(1, s) - 5;
            omega_mu_1_unscaled = [signal_beta_1, repelem(baseline, 1,p-s)];
            omega_mu_2_unscaled = [signal_beta_2, repelem(baseline, 1,p-s)];
            beta_unscaled = (omega_mu_1_unscaled - omega_mu_2_unscaled);
            sep_scale = sqrt(beta_unscaled * Sigma * beta_unscaled');            M = sep /sep_scale ;
            omega_mu_1 = M*omega_mu_1_unscaled;
            omega_mu_2 = M*omega_mu_2_unscaled;
            mu1 = Omega_star \ omega_mu_1';
            mu2 = Omega_star \ omega_mu_2';
            beta_star = omega_mu_1 - omega_mu_2;
    end
    
    
    % Mahalanobis distance
    mahala_dist_sq = (mu1 - mu2)' * Omega_star * (mu1 - mu2);
    mahala_dist = sqrt(mahala_dist_sq);
    %fprintf('Mahalanobis distance between mu1 and mu2: %.4f\n', mahala_dist);
    % Generate noise once
    rng(seed);
    Z = mvnrnd(zeros(p, 1), Sigma, n);  % n x p noise
    % Create mean matrix
    mean_matrix = [repmat(mu1', n1, 1); repmat(mu2', n2, 1)];
    % Final data matrix
    X = Z + mean_matrix;
end
%% 
%% 
%% 
%% 
%% 
