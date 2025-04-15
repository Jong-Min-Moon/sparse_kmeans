function Omega_star = get_precision_ER(p)
%% get_precision_ER
% @export
% 
% Generate a precision matrix representing an Erdős–Rényi Random Graph, using 
% vectorized operations (no explicit for-loops). The random seed is fixed as |rng(1)| 
% for reproducibility. The generation procedure is as follows:
%% 
% * Let $ \tilde{\Omega} = (\tilde{\omega}_{ij})$ where 
% * $\tilde{\omega}_{ij} = u_{ij} \delta_{ij},$ where
% * $\delta_{ij} \sim \text{Bernoulli}(0.05)$ is a Bernoulli random variable 
% with success probability 0.05, 
% * $u_{ij} \sim \text{Uniform}([0.5, 1] \cup [-1, -0.5])$
% * After symmetrizing $\tilde{\Omega}$, to ensure positive definiteness, define:
% * $\Omega^* = \tilde{\Omega} + \left\{ \max\left(-\phi_{\min}(\tilde{\Omega}), 
% 0\right) + 0.05 \right\} I_p$.
% * Finally, $\Omega^*$ is standardized to have unit diagonals.
%% 
% 
    rng(1);  % set random seed for reproducibility
    % Get upper triangle indices (excluding diagonal)
    upper_idx = triu(true(p), 1);
    % Total number of upper triangle entries
    num_entries = sum(upper_idx(:));
    % Generate Bernoulli mask: 1 with probability 0.05
    mask = rand(num_entries, 1) < 0.05;
    % Generate random values from Unif([-1, -0.5] ∪ [0.5, 1])
    signs = 2 * (rand(num_entries, 1) < 0.5) - 1;   % ±1 with equal prob
    mags  = rand(num_entries, 1) * 0.5 + 0.5;       % ∈ [0.5, 1]
    values = signs .* mags;
    % Apply mask to only keep active edges
    values(~mask) = 0;
    % Build upper triangle of matrix
    Omega_tilde = zeros(p);
    Omega_tilde(upper_idx) = values;
    % Symmetrize
    Omega_tilde = Omega_tilde + Omega_tilde';
    % Ensure positive definiteness
    min_eig = min(eig(Omega_tilde));
    delta = max(-min_eig, 0) + 0.05;
    Omega_star = Omega_tilde + delta * eye(p);
    % Standardize to have unit diagonal
    D_inv = diag(1 ./ sqrt(diag(Omega_star)));
    Omega_star = D_inv * Omega_star * D_inv;
end
%% 
% 
