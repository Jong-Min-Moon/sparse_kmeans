function [mean_now, noise_now, Omega_diag_hat] = ISEE_bicluster(dg)
    p = dg.dimension
    n = dg.sample_size;
    n_regression = floor(p/2);
    cluster_est_now = dg.cluster_info_vec;

    Omega_diag_hat_even = repelem(0,p/2);
    Omega_diag_hat_odd = repelem(0,p/2);
    Omega_diag_hat = repelem(0,p);

    mean_now_even = zeros(p/2,n);
    mean_now_odd = zeros(p/2,n);
    mean_now = zeros(p,n);

    noise_now_even =zeros(p/2,n);
    noise_now_odd = zeros(p/2,n);
    noise_now = zeros(p,n);


    parfor i = 1 : n_regression
        alpha_Al = zeros([2,2]);
        E_Al = zeros([2,n]);

        for cluster = 1:2
            g_now = (cluster_est_now == cluster);
            x_noisy_g_now = dg.data(:,g_now);
            predictor_boolean = ((1:p) == (2*(i-1)+1)) | ((1:p) == (2*(i-1)+2));
            predictor_now = x_noisy_g_now(~predictor_boolean, :)';
            for j = 1:2
                boolean_now = (1:p) == (2*(i-1)+j);
                response_now = x_noisy_g_now(boolean_now,:)';
                model_lasso = glm_gaussian(response_now, predictor_now); 
                fit = penalized(model_lasso, @p_lasso, "standardize", true);
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
        Omega_hat_Al = inv(E_Al*E_Al')*n;% 2 x 2
        diag_Omega_hat_Al = diag(Omega_hat_Al);
        noise_Al = Omega_hat_Al*E_Al; % 2 * n
        mean_Al = zeros([2,n]);
        for cluster = 1:2
            g_now = cluster_est_now == cluster;
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



    even_idx =mod((1:p),2)==0;
    odd_idx = mod((1:p),2)==1;

    Omega_diag_hat(odd_idx) = Omega_diag_hat_odd;
    Omega_diag_hat(even_idx) = Omega_diag_hat_even;


    mean_now(odd_idx,:) = mean_now_odd;
    mean_now(even_idx,:) = mean_now_even;
    noise_now(odd_idx,:) = noise_now_odd;
    noise_now(even_idx,:) = noise_now_even;

    Omega_diag_hat = Omega_diag_hat';

