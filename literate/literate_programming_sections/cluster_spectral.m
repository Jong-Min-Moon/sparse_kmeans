%% cluster_spectral
% @export
% 
% Inputs
%% 
% * x: $p\times n$data matrix, where $p$ is the data dimension and $n$ is the 
% sample size
% * k: positive integer. number of cluster.
%% 
% Outputs
%% 
% * cluster_est: n array of positive integers, where n is the sample size. ex. 
% [1 2 1 2 3 4 2 ]
function cluster_est = cluster_spectral(x, k)
    n = size(x,2);
    p = size(x,1);
    H_hat = (x' * x)/n; %n x n  affinity matrix
    [V,D] = eig(H_hat);
    [d,ind] = sort(diag(D), "descend");
        Ds = D(ind,ind);
        
    Vs = V(:,ind);
    tau_n = 1/ log(n+p);
    delta_n = tau_n^2;
        f1 = abs(sum(V(:,1)))/sqrt(n) - 1;
    if d(1)/d(2) < 1+ tau_n
        new_data = V(:,1:2);
    elseif f1 > delta_n
        new_data = V(:,1);
    else
        new_data = V(:,2);
    end
    [cluster_est,~] = kmeans(new_data,k);
    cluster_est= cluster_est';
end
%% 
% 
% 
% We begin by implementing a single step of the algorithm, which we then use 
% to construct the full iterative procedure. Each step consists of two components: 
% variable selection and SDP-based clustering. We implement these two parts sequentially 
% and combine them into a single step function.
%% 
%%  
