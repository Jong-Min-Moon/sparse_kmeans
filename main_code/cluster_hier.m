%% cluster_spectral
% @export
function cluster_est = cluster_hier(x, k)
    n = size(x,2);
    H_hat = (x' * x)/n; %compute affinity matrix
    [V,D] = eig(H_hat);
    [d,ind] = sort(diag(D), "descend");
    Ds = D(ind,ind);
    Vs = V(:,ind);
    [cluster_est,C] = kmeans(Vs(:,1),k);
    cluster_est= cluster_est';
end
