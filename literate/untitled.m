%% kmeans_sdp_pengwei
% The following implementation, originally written by Mixon, Villar, and Ward 
% and last edited on January 20, 2024, uses SDPNAL+ [3] to solve the Peng and 
% Wei k-means SDP formulation [2], following the approach described in [1]. The 
% original version accepts a $p\times n$ data matrix as input. To accommodate 
% both isotropic and non-isotropic cases, we modify the code to accept an affinity 
% matrix instead. Given an affinity matrix $A$ , The code solves the following 
% problem:
% 
% $\hat{Z} = \arg\max_{Z \in \mathbb{R}^{n \times n}} \langle A, Z \rangle \quad 
% \text{subject to} \quad Z \succeq 0,\; \mathrm{tr}(Z) = K,\; Z \mathbf{1}_n 
% = \mathbf{1}_n,\; Z \geq 0$.
%% 
% * Inputs:
% * A : A n x n array of affinity matrix where n denotes the number of observations. 
% * k: The number of clusters.
% * Outputs:
% * X:  A N x N array corresponding to the solution of Peng and Wei's solution.            
%% 
% References:
%% 
% # Mixon, Villar, Ward. Clustering subgaussian mixtures via semidefinite programming
% # Peng, Wei. Approximating k-means-type clustering via semidefinite programming
% # Yang, Sun, Toh. Sdpnal+: a majorized semismooth newton-cg augmented lagrangian 
% method for semidefinite programming with nonnegative constraints


function X=kmeans_sdp_pengwei(A, k)

% old version: Construction of distance squared matrix.

%N=size(P,2);
% D = -P'*P;

% new version:
D = -A;
N=size(P,2);


% SDP definition for SDPNAL+
n=N;
C{1}=D;
blk{1,1}='s'; blk{1,2}=n;
b=zeros(n+1,1);
Auxt=spalloc(n*(n+1)/2, n+1, 5*n);
Auxt(:,1)=svec(blk(1,:), eye(n),1);

b(1,1)=k;
idx=2;
for i=1:n
    A=zeros(n,n);
    A(:,i)=ones(n,1);
    A(i,:)=A(i,:)+ones(1,n);
    b(idx,1)=2;
    Auxt(:,idx)= svec(blk(1,:), A,1);
    idx=idx+1;
end
At{1}=sparse(Auxt);

OPTIONS.maxiter = 50000;
OPTIONS.tol = 1e-6;
OPTIONS.printlevel = 0;

% SDPNAL+ call
[obj,X,s,y,S,Z,y2,v,info,runhist]=sdpnalplus(blk,At,C,b,0,[],[],[],[],OPTIONS);

X=cell2mat(X);

end

end
%% 
%% testing
% dfdf

% Load or create your affinity matrix A (example: from Gaussian RBF)
rng(1);
X = [randn(50,2)*0.75 + [2 2]; randn(50,2)*0.5 + [-2 -2]; randn(50,2)*0.6 + [2 -2]];
n = size(X,1);
k = 3;

% Build Gaussian kernel affinity matrix
sigma = 1.0;
A = exp(-pdist2(X, X).^2 / (2*sigma^2));

% Run Peng-Wei SDP k-means
Z = kmeans_sdp_pengwei(A, k);

% Extract clustering: use k-means on top eigenvectors
[U,~,~] = svd(Z); 
U_k = U(:,1:k);  % top-k eigenvectors
clusters = kmeans(U_k, k, 'Replicates',10);

% Visualization
figure;
gscatter(X(:,1), X(:,2), clusters, 'rgb', 'o', 8);
title('Clustering result from Peng-Wei SDP relaxation');
axis equal;

%% Function 2
% 
% Implementation of the stopping criteria
% 
% K means objective function

%function objective_value_original = get_objective_value_original(x, cluster_est)
   %     objective_value_original = 0;
  %      for i = 1:ik.data_object.number_cluster
  %          cluster_size = sum(cluster_est==i);
 %          affinity_cluster = ik.data_object.sparse_affinity(cluster_est==i, cluster_est==i);
 %           within_cluster_variation = ((-2*sum(affinity_cluster, "all") + 2*cluster_size*trace(affinity_cluster))/cluster_size;
 %           objective_value_original = objective_value_original + within_cluster_variation;
 %       end

   % end