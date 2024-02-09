function [Theta] = glasso(S,rho)
    [n,p] = size(S);
    max_iterations = 100;
    t = 1e-4;
    convergence_value = t * meanabs(S - diag(diag(S)));

    % initialise
    W_old = S + rho*eye(p);
    W = W_old;

    for round=1:max_iterations
        for j=p:-1:1
            i = j;

            W11 = W;
            W11(i,:) = [];  % remove ith row
            W11(:,j) = [];  % remove jth column
            w22 = W(i,j);

            s12 = S(:,j);
            s12(i,:) = [];
            
            A = W11^0.5;
            b = (W11^-0.5)*s12;
            
            beta = chenLasso(A,b,rho,1e2,1e-4);
            w12 = W11 * beta;

            W_left = W11(:,1:j-1);
            W_right = W11(:,j:p-1);
            W = [W_left w12 W_right];

            w12_row = [w12(1:j-1) ; w22 ; w12(j:p-1)]';
            W_above = W(1:i-1,:);
            W_below = W(i:p-1,:);
            W = [W_above ; w12_row ; W_below];
        end
        if meanabs(W - W_old) < convergence_value
           break; 
        end
        W_old = W;
    end
Theta = W^-1;



function b = chenLasso(X, Y, lambda, maxIt, tol),
% an algorithm to solve the lasso problem
% from http://pages.stat.wisc.edu/~mchung/teaching/768/matlab/CS/graphicalLasso.m

if nargin < 4, tol = 1e-6; end
if nargin < 3, maxIt = 1e2; end

% Initialization
[n,p] = size(X);
if p > n,
    b = zeros(p,1); % From the null model, if p > n
else
    b = X \ Y;  % From the OLS estimate, if p <= n
end
b_old = b;
i = 0;

% Precompute X'X and X'Y
XTX = X'*X;
XTY = X'*Y;

% Shooting loop
while i < maxIt,
    i = i+1;
    for j = 1:p,
        jminus = setdiff(1:p,j);
        S0 = XTX(j,jminus)*b(jminus) - XTY(j);  % S0 = X(:,j)'*(X(:,jminus)*b(jminus)-Y)
        if S0 > lambda,
            b(j) = (lambda-S0) / norm(X(:,j),2)^2;
        elseif S0 < -lambda,
            b(j) = -(lambda+S0) / norm(X(:,j),2)^2;
        else
            b(j) = 0;
        end
    end
    delta = norm(b-b_old,1);    % Norm change during successive iterations
    if delta < tol, break; end
    b_old = b;
end
if i == maxIt,
    fprintf('%s\n', 'Maximum number of iteration reached, shooting may not converge.');
end