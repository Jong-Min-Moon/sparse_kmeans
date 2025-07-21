%% kmeans_sdp_pengwei
% @export
% 
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
% 
% Inputs:
%% 
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
D = -A;
N=size(A,2);
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
