function cluster_est_now = initialize_bicluster(x, init_method) 
    K = 2;
    n = size(x,2);
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
    %cluster_est_now = cluster_est_now .* (cluster_est_now ~= 2) + (cluster_est_now == 2)* (-1);
    cluster_est_now = cluster_est_now';