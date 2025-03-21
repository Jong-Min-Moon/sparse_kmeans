classdef sdp_kmeans < handle
    % Author:       Mixon, Villar, Ward.
    % Filename:     kmeans_sdp.m
    % Last edited:  2024-01-20 
    % Description:  This class uses SDPNAL+ [3] to solve the Peng and Wei's 
    %               k-means SDP [2], according to the formulation in [1].

    properties
        X           % Data matrix (d x n)
        K           % Number of clusters
        n           % Number of data points
        D           % Distance matrix (n x n)
        Z_opt       % SDP solution
        cluster_est % Estimated cluster labels (1 x n)
    end

    methods
        function obj = sdp_kmeans(X, K)
            % Constructor
            if nargin < 2
                error('Two input arguments required: data matrix X and number of clusters K.');
            end
            if ~ismatrix(X) || ~isnumeric(X)
                error('Input X must be a numeric matrix.');
            end
            if ~isscalar(K) || K <= 1 || K ~= floor(K)
                error('Number of clusters K must be an integer greater than 1.');
            end

            obj.X = X;
            obj.K = K;
            obj.n = size(X, 2);

            if obj.K > obj.n
                error('Number of clusters K cannot exceed number of data points.');
            end

            obj.D = -X' * X;
        end

        function cluster_est = fit_predict(obj)
            % Run full SDP and clustering pipeline
            obj.run_SDP();
            obj.sdp_to_cluster();
            cluster_est = obj.cluster_est;
        end

        function run_SDP(obj)
            % Run Peng–Wei SDP via SDPNAL+
            n = obj.n;
            D = obj.D;
            k = obj.K;

            C{1} = D;
            blk{1,1} = 's'; blk{1,2} = n;
            b = zeros(n+1, 1);
            Auxt = spalloc(n*(n+1)/2, n+1, 5*n);
            Auxt(:,1) = svec(blk(1,:), eye(n), 1);
            b(1,1) = k;

            idx = 2;
            for i = 1:n
                A = zeros(n, n);
                A(:, i) = ones(n, 1);
                A(i, :) = A(i, :) + ones(1, n);
                b(idx, 1) = 2;
                Auxt(:, idx) = svec(blk(1,:), A, 1);
                idx = idx + 1;
            end

            At{1} = sparse(Auxt);

            OPTIONS.maxiter = 50000;
            OPTIONS.tol = 1e-6;
            OPTIONS.printlevel = 0;

            % Call SDPNAL+
            [~, X, ~, ~, ~, ~, ~, ~, ~, ~] = sdpnalplus(blk, At, C, b, 0, [], [], [], [], OPTIONS);
            obj.Z_opt = cell2mat(X);
        end

        function sdp_to_cluster(obj)
            % Recover cluster labels from SDP solution via spectral method
            [U_sdp, ~, ~] = svd(obj.Z_opt);
            U_top_k = U_sdp(:, 1:obj.K);
            cluster_labels = kmeans(U_top_k, obj.K);  % returns column vector
            obj.cluster_est = cluster_labels';        % convert to row vector
        end
    end
end
