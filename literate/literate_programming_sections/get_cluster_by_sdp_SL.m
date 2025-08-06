%% get_cluster_by_sdp_SL
% @export
function cluster_est = get_cluster_by_sdp_SL(X,K) 
    n = size(X,2); % Sample size
    p = size(X,1); % dimension
    gama = 0.1;
    columns=(rand(1,n) <gama );
    q =sum(columns);  % Random select q data points
    X_hat = X(:,columns); % New matrix with dimension p*q    
    idx_hat=get_cluster_by_sdp(X_hat, K);
 
    sumsub = histcounts(idx_hat, 1:K+1);
    C_hat=zeros(p,K);
    X_hat_1=X_hat';
 
    % Get the centers
    for cc=1:K
        findindx=find(idx_hat==cc);
        newcoln=randperm(sumsub(cc),min(sumsub));
        newindx=findindx(newcoln);
        linearIndices = newindx;
        inter=mean(X_hat_1(linearIndices,:));
        C_hat(:,cc)=inter';
    end
  
    % Assign Xi to nearesr centroid of X_hat
    cluster_est = zeros(n,1);
        for j=1:n
            fmv=zeros(1,K);
            for i=1:K
                fmv(1,i)=norm(X(:,j)-C_hat(:,i)); % Every point compared with centers
            end
            [mv,mp]=min(fmv);
        cluster_est(j)=mp; % Assigned to the position of center
    end
cluster_est = cluster_est';
end 
