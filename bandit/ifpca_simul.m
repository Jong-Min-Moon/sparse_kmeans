classdef ifpca_simul  < handle

    properties
        X           % Data matrix (d x n)
        K           % Number of clusters
        n           % Number of data points
        p           % Data dimension
        cutoff      % Threshold for variable inclusion
        alpha       % Alpha parameters of Beta prior
        beta        % Beta parameters of Beta prior
        pi
        acc_dict
        cluster_est_dict
        signal_entry_est
        n_iter

        x_tilde_est       
        omega_est_time    
        sdp_solve_time    
        entries_survived  
        obj_val_prim     
        obj_val_dual      
        obj_val_original  
    end

    methods
        function obj = ifpca_simul(X, number_cluster)
            obj.X =X;
            obj.K = number_cluster;
            %obj.omega_sparsity = omega_sparsity;

            obj.n = size(obj.X, 2);
            obj.p = size(obj.X, 1);


            
        end       
        function initialize_cluster_est(obj)
            cluster_est_dummy   = cluster_est( repelem(1,obj.n) );
            obj.cluster_est_dict = repelem(cluster_est_dummy, 1); %dummy
            obj.acc_dict = NaN;
        end

        function fit_predict(obj, cluster_true)
            obj.n_iter = 0;
            [label, stats, numselect] = ifpca_original(obj.X, obj.K);
            label = label';
            cluster_est_label = cluster_est(label);
            obj.cluster_est_dict = cluster_est_label
            obj.evaluate_accuracy(cluster_true);
        end
        


        function initialize_saving_matrix(obj)
            obj.x_tilde_est       = zeros(obj.p, obj.n, 1);
            obj.omega_est_time    = zeros(1, 1);
            obj.sdp_solve_time    = zeros(1, 1);
            obj.entries_survived  = zeros(1, obj.p);
            obj.obj_val_prim      = zeros(1, 1);
            obj.obj_val_dual      = zeros(1, 1);
            obj.obj_val_original  = zeros(1, 1);
        end

        function evaluate_accuracy(obj, cluster_true)
            cluster_est_now = obj.cluster_est_dict;
            obj.acc_dict = cluster_est_now.evaluate_accuracy(cluster_true);
            obj.acc_dict
        end % end of method evaluate_accuracy



        function database_subtable = get_database_subtable(obj, rep, Delta, rho, support, cluster_true, Omega)
            s = length(support);
            current_time = get_current_time();
            %[diff_x_tilde_fro, diff_x_tilde_op, diff_x_tilde_ellone] = obj.evaluate_innovation_est(Omega);

            %fprintf( strcat( "acc =", join(repelem("%f ", length(acc_vec))), "\n"),  acc_vec );
            
            
            cluster_string_dict = obj.get_cluster_string_dict();
            
            %values(obj.acc_dict);
            %values(cluster_string_dict);
             
            n_row = 1;

            database_subtable = table(...
                repelem(rep, n_row)',...                      % 01 replication number
                [1],...                                  % 02 step iteration number
                repelem(Delta, n_row)',...                    % 03 separation
                repelem(obj.p, n_row)',... % 04 data dimension
                repelem(rho  , n_row)',...                      % 05 conditional correlation
                repelem(s    , n_row)',...                        % 06 sparsity
                ...
                repelem(false, n_row)',... %07
                repelem(false, n_row)',...      %08
                repelem(false, n_row)',...     %09
                [obj.acc_dict],...                                     % 10 accuracy
                ...
                repelem(0, n_row)',...               % 11 objective function value (relaxed, primal)
                repelem(0, n_row)',...               % 12 objective function value (relaxed, dual)
                repelem(0, n_row)',...           % 13 objective function value (original)
                ...
                [0],...                           % 14 true positive
                [0],...                          % 15 false positive
                [0],...                          % 16 false negative
                ...
                repelem(0, n_row)',...                       % 13 estimation error of the innovated data, in Frobenius norm
                repelem(0, n_row)',...                        % 14 estimation error of the innovated data, in operator norm
                repelem(0, n_row)',...                    % 15 estimation error of the innovated data, in \ell_1 norm
                repelem(0, n_row)',...             % 16 timing for estimating the precision matrix
                repelem(0, n_row)', ...            % 17 timing elapsed for solving the SDP
                repelem(current_time, n_row)', ...            % 18 timestamp
                [""],...                      % 19 indices of survived entry
                string(cluster_string_dict)',...                          % 20 clustering information
                'VariableNames', ...
                ...  1      2       3      4      5        6         
                ["rep", "iter", "sep", "dim", "rho", "sparsity", ...
                ...
                "stop_og", "stop_sdp", "stop_loop", ...
                ...  7        8           9             10             
                 "acc", "obj_prim", "obj_dual", "obj_original", ...
                ...11               12
                 "true_pos", "false_pos",  "false_neg"...
                ...       13              14                    15
                 "diff_x_tilde_fro", "diff_x_tilde_op", "diff_x_tilde_ellone", ...
                ...  16          17           18
                 "time_est", "time_SDP", "jobdate", ...
                ...      19              20            
                "survived_indices", "cluster_est"
                ]);
        end % end of get_database_subtable



        function cluster_string_vec = get_cluster_string_dict(obj)   
            cluster_est_now = obj.cluster_est_dict;
            cluster_string_vec = cluster_est_now.cluster_info_string;
        end% end of cluster_string_vec   
     
    
    end % end of method
end % end of class