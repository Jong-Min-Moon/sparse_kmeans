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
        A_mat=zeros(n,n);
        A_mat(:,i)=ones(n,1);
        A_mat(i,:)=A_mat(i,:)+ones(1,n);
        b(idx,1)=2;
        Auxt(:,idx)= svec(blk(1,:), A_mat,1);
        idx=idx+1;
    end
    At{1}=sparse(Auxt);
    OPTIONS.maxiter = 50000;
    OPTIONS.tol = 1e-6;
    OPTIONS.printlevel = 0;
    
    % Check if sdpnalplus exists
    if exist('sdpnalplus', 'file')
        % SDPNAL+ call
        [obj,X,s,y,S,Z,y2,v,info,runhist]=sdpnalplus(blk,At,C,b,0,[],[],[],[],OPTIONS);
        X=cell2mat(X);
    else
        % Fallback or error?
        % Assuming user has it. If not, we might need a mock for this simulation?
        % Let's assume it exists as per user context.
        error('SDPNAL+ solver (sdpnalplus) not found in path.');
    end
end
