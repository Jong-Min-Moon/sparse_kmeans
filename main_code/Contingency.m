%% Contingency
% @export
% 
% 
% 
% Function Signature
% 
% C = Contingency(c1, c2)
% 
% *Description*
% 
% Computes the contingency matrix C for two clusterings, which is the foundational 
% matrix for calculating various cluster similarity indices. This function is 
% automatically called by RandIndex.
% 
% |*Input Arguments*|
%% 
% *Argument*
%% 
% *Description*
%% 
% *Required Format*
%% 
% c1
%% 
% Vector of cluster assignments (intergers) for N points.
%% 
% Vector (Row or Column) of equal length to c2.
%% 
% c2
%% 
% Vector of cluster assignments (intergers) for N points.
%% 
% Vector (Row or Column) of equal length to c1.
%% 
% |*Output Arguments*|
%% 
% *Output*
%% 
% *Description*
%% 
% *Format*
%% 
% C
%% 
% The Contingency matrix. C(i, j) is the count of data points assigned to the 
% i-th unique cluster in c1 and the j-th unique cluster in c2.
%% 
% K1 x K2 matrix, where K1 and K2 are the number of unique clusters in c1 and 
% c2, respectively.
%% 
% |*Implementation Note*|
% 
% This implementation uses unique() to map the original cluster labels (which 
% can be arbitrary integers) to sequential 1-based indices for the matrix rows 
% and columns, ensuring the matrix size is exactly (# unique clusters in c1) x 
% (# unique clusters in c2)
% 
% 
% 
% (C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk
% 
% This code is taken directly from https://github.com/drjingma/gmm and has not 
% been modified. 
% 
% 
function Cont=Contingency(Mem1,Mem2)
if nargin < 2 || min(size(Mem1)) > 1 || min(size(Mem2)) > 1
   error('Contingency: Requires two vector arguments')
end
Cont=zeros(max(Mem1),max(Mem2));
for i = 1:length(Mem1);
   Cont(Mem1(i),Mem2(i))=Cont(Mem1(i),Mem2(i))+1;
end
%% 
% 
% 
% 
%% 
%% 
% 
