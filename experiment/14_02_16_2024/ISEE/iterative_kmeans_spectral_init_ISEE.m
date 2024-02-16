function [cluster_acc, diff_x_tilde, diff_omega_diag, false_discov, true_discov, false_discov_top5, omega_est_time, sdp_solve_time] = iterative_kmeans_spectral_init_ISEE(x, K, n_iter, Omega, s, cluster_true, init_method, verbose, sdp_method) 
% Sigma = UNknown covariance matrix
%data generation
% created 01/26/2024
init_method
sdp_method


% spectral initialization
n = size(x,2);
p = size(x,1);
thres = sqrt(2 * log(p) )
fprintf("thres=%f", thres)
Omega_x = Omega*x;
diff_x_tilde = zeros(1,n_iter);
diff_omega_diag = zeros(1,n_iter);
false_discov = zeros(1,n_iter);
true_discov = zeros(1,n_iter);
false_discov_top5 = repelem("0", n_iter);
omega_est_time = zeros(1,n_iter);
sdp_solve_time = zeros(1,n_iter);

if strcmp(init_method, 'spec')
    H_hat = (x' * x)/n;
    [V,D] = eig(H_hat);
    [d,ind] = sort(diag(D), "descend");
    Ds = D(ind,ind);
    Vs = V(:,ind);
    [cluster_est_now,C] = kmeans(Vs(:,1),K);
elseif strcmp(init_method, "hc")
    Z = linkage(x', 'ward');
    cluster_est_now = cluster(Z, 'Maxclust',K);
elseif strcmp(init_method, "SDP")
    Sigma_est_now = cov(x');
    X_tilde_now = linsolve(Sigma_est_now, x);
    Z_now = kmeans_sdp( x'* X_tilde_now/ n, K);       
    
    % final thresholding
    [U_sdp,~,~] = svd(Z_now);
    U_top_k = U_sdp(:,1:K);
    [cluster_est_now,C] = kmeans(U_top_k,K);  % label

end
cluster_est_now = cluster_est_now .* (cluster_est_now ~= 2) + (cluster_est_now == 2)* (-1);
cluster_est_now = cluster_est_now';

cluster_acc_before_thres = max( mean(cluster_true ==  cluster_est_now), mean(cluster_true == -cluster_est_now));



n_g1_now = sum(cluster_est_now == 1);
n_g2_now = sum(cluster_est_now ==-1);

if verbose
    fprintf("\np = %i, acc_init: %f \n", p, cluster_acc_before_thres);
    fprintf("n_{G1}_init = %i, n_{G1}_init = %i\n", n_g1_now, n_g2_now )
    fprintf("threshold: (%f)\n", thres)
    
end
cluster_acc_now = cluster_acc_before_thres;

for iter = 1:n_iter
    if verbose
        fprintf("\n%i th thresholding\n\n", iter)
    end
    % 1. estimate cluster means


    if max(n_g1_now, n_g2_now) == n
        %fprintf("all observations are clustered into one group")
        cluster_acc = 0.5;
        return 
    end
    
    cluter_code = [1,-1];
n_regression = floor(p/2)
Omega_diag_hat_even = repelem(0,p/2);
Omega_diag_hat_odd = repelem(0,p/2);
mean_now_even = repelem(p/2,n);
mean_now_odd = repelem(p/2,n);
noise_now_even = repelem(p,n);
noise_now_odd = repelem(p,n);
tic
parfor i = 1 : n_regression
    alpha_Al = zeros([2,2]);
    E_Al = zeros([2,n]);

    for cluster = 1:2
        clutser_code_now = cluter_code(cluster);
        g_now = cluster_est_now == clutser_code_now;
        x_noisy_g_now = x(:,g_now);
        for j = 1:2
            boolean_now = (1:p) == (2*(i-1)+j);
            response_now = x_noisy_g_now(boolean_now,:)';
            predictor_now = x_noisy_g_now(~boolean_now, :)';
            model_lasso = glm_gaussian(response_now, predictor_now); 
            fit = penalized(model_lasso, @p_lasso);
            AIC = goodness_of_fit('aic', fit);
            [min_aic, min_aic_idx] = min(AIC);
            beta = fit.beta(:,min_aic_idx);
            slope = beta(2:end);
            intercept = beta(1);
            E_Al(j,g_now) = response_now - intercept- predictor_now * slope;
            alpha_Al(j, cluster) = intercept;
        end

        
        
        
        
    end
    %estimation
    Omega_hat_Al = inv(E_Al*E_Al')*n; % 2 x 2
    diag_Omega_hat_Al = diag(Omega_hat_Al);
    noise_Al = Omega_hat_Al*E_Al; % 2 * n
    mean_Al = zeros([2,n]);
    for cluster = 1:2
        clutser_code_now = cluter_code(cluster);
        g_now = cluster_est_now == clutser_code_now;
        n_now = sum(g_now);
        mean_Al(:,g_now) = repmat(Omega_hat_Al*alpha_Al(:,cluster), [1,n_now]);
    end
    %Omega_diag_hat( output_index ) = diag(Omega_hat_Al);
    k = i+1;
    Omega_diag_hat_odd( i ) = diag_Omega_hat_Al(1);
    Omega_diag_hat_even( i) = diag_Omega_hat_Al(2);
    mean_now_odd( i,:) = mean_Al(1,:);
    mean_now_even( i,:) = mean_Al(2,:);
    noise_now_odd( i,:) = noise_Al(1,:);
    noise_now_even( i,:) = noise_Al(2,:);
end
toc
even_idx =mod((1:p),2)==0;
odd_idx = mod((1:p),2)==1;
Omega_diag_hat = repelem(0,p);
Omega_diag_hat(odd_idx) = Omega_diag_hat_odd;
Omega_diag_hat(even_idx) = Omega_diag_hat_even;
diff_omega_diag(iter) = norm(Omega_diag_hat - diag(Omega))

mean_now = repelem(p,n);
mean_now(odd_idx,:) = mean_now_odd;
mean_now(even_idx,:) = mean_now_even;
noise_now(odd_idx,:) = noise_now_odd;
noise_now(even_idx,:) = noise_now_even;
x_tilde_now = mean_now + noise_now;
diff_x_tilde(iter) = norm(x_tilde_now-Omega_x, "fro")
    % 2. threshold the data matrix

    signal_est_now = mean( mean_now(:, cluster_est_now==1), 2) - mean( mean_now(:, cluster_est_now==-1), 2);
    abs_diff = signal_est_now;
    thres = 0.1;

    %signal_est_now = mean( x_tilde_now(:, cluster_est_now==1), 2) - mean( x_tilde_now(:, cluster_est_now==-1), 2);   
    %abs_diff = abs(signal_est_now')./sqrt(Omega_diag_hat) * sqrt( n_g1_now*n_g2_now/n );
    [abs_diff_sort, abs_diff_sort_idx]= sort(abs_diff, "descend");
    discov_idx_sorted = abs_diff_sort_idx(abs_diff_sort>thres);
    false_discov_idx_sorted = discov_idx_sorted(discov_idx_sorted>s);

    s_hat = abs_diff > thres;
    n_entries_survived = sum(s_hat);
        entries_survived = find(s_hat);


    if n_entries_survived == 0
        disp("no entry survived")
        cluster_acc = 0.5;
        break
    end

    tic
    %estimate sigma hat s
    X_g1_now = x(:, (cluster_est_now ==  1)); 
    X_g2_now = x(:, (cluster_est_now ==  -1)); 
    X_mean_g1_now = mean(X_g1_now, 2);
    X_mean_g2_now = mean(X_g2_now, 2);
    data_py = [(X_g1_now - X_mean_g1_now) (X_g2_now - X_mean_g2_now)]';
    data_filtered = data_py(:,s_hat);
    Sigma_hat_s_hat_now = data_filtered' * data_filtered/(n-1);


 
    x_tilde_now_s  = x_tilde_now(s_hat,:);  
    x_tilde_est_time(iter) = toc;

    tic
    Z_now = kmeans_sdp( x_tilde_now_s' * Sigma_hat_s_hat_now * x_tilde_now_s/ n, K);
    sdp_solve_time(iter) = toc
    
    % final thresholding
    [U_sdp,~,~] = svd(Z_now);
    U_top_k = U_sdp(:,1:K);
    [cluster_est_now,C] = kmeans(U_top_k,K);  % label
    cluster_est_now = cluster_est_now .* (cluster_est_now ~= 2) + (cluster_est_now == 2)* (-1);    
    cluster_est_now = cluster_est_now';   
    cluster_acc_now = max( ...
                    mean(cluster_true == cluster_est_now), ...
                    mean(cluster_true == -cluster_est_now) ...
                    );
    n_g1_now = sum(cluster_est_now == 1);
    n_g2_now = sum(cluster_est_now ==-1);

    top_num = min(5, length(false_discov_idx_sorted));
    if top_num > 0
        false_discov_top5(iter) = strjoin(arrayfun(@(x) num2str(x), false_discov_idx_sorted(1:top_num) ,'UniformOutput',false),'_')
    end
    
        fprintf("right : (%i)\n", sum(s_hat(1:s)))
    fprintf("wrong : (%i)\n", sum(s_hat(s+1:end)))  
    false_discov(iter) = sum(s_hat(s+1:end));
    true_discov(iter) = sum(s_hat(1:s));
    
    if verbose
        fprintf("\n%i entries survived \n",n_entries_survived)

        %fprintf("normalized difference top 10 max: (%f)\n", top_10)
        %fprintf("normalized difference top 10 max index: (%i)\n", top_10_idx)
        fprintf("n_{G1}_now = %i, n_{G1}_now = %i\n", n_g1_now, n_g2_now )
        fprintf("acc_now= %f", cluster_acc_now);

    end
    % end one iteration
end % end of iterative algorithm
cluster_acc = cluster_acc_now
diff_x_tilde
diff_omega_diag
false_discov
true_discov
false_discov_top5
omega_est_time
sdp_solve_time
