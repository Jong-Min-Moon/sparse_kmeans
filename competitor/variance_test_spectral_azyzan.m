classdef variance_test_spectral_azyzan < handle

    properties
        X           % Data matrix (d x n)
        K           % Number of clusters
        n           % Number of data points
        p           % Data dimension

        var_coordinatewise         % p-array: coordinate-wise variance
        var_min
        alpha       % variance scaling

        acc_dict
        cluster_est_dict
        signal_entry_est

        x_tilde_est       
        omega_est_time    
        sdp_solve_time    
        entries_survived  
        obj_val_prim     
        obj_val_dual      
        obj_val_original  
    end

    methods
        function obj = variance_test_spectral_azyzan(X)

            if ~ismatrix(X) || ~isnumeric(X)
                error('Input X must be a numeric matrix.');
            end


            obj.X = X;
            obj.K = 2;
            obj.n = size(X, 2);
            obj.p = size(X, 1);
            obj.var_coordinatewise = repmat(NaN,1,obj.p);

        end % end of initialization




        function signal_entry_est = get_signal_entry_est(obj)
            %If A is a matrix whose columns are random variables and whose rows are observations, then V is a row vector containing the variance corresponding to each column.
            obj.var_coordinatewise  = var(obj.X');
            thres = obj.get_threshold();
            signal_entry_est =  obj.var_coordinatewise > thres;
            obj.signal_entry_est = signal_entry_est;
        end % end of method get_signal_entry_est


        function threshold = get_threshold (obj)
            obj.var_min = min(obj.var_coordinatewise);
            log_np = log( obj.n * obj.p);
            alpha = sqrt(6 * ( log_np / obj.n)) + (2 * log_np / obj.n);
            threshold = obj.var_min * (1+alpha)/(1-alpha);
        end


        function fit_predict(obj)
            obj.get_signal_entry_est();
            X_sub = obj.X(obj.signal_entry_est,:);
            mean_sub = mean(X_sub')';
            sample_cov_sub = cov(X_sub');
            [V, D] = eig(sample_cov_sub);
            V
            v1 = V(:,end)
            cluster_est = (X_sub-mean_sub)' * v1 > 0

            

        end % end of fit_predict
        
   
    end % end of methods
end
