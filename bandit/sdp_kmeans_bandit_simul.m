classdef sdp_kmeans_bandit_simul  < sdp_kmeans_bandit 
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

        data_object
        init_method
        omega_sparsity
        x_tilde_est       
        omega_est_time    
        sdp_solve_time    
        entries_survived  
        obj_val_prim     
        obj_val_dual      
        obj_val_original  
    end

    methods
        function obj = iterative_kmeans(data_object, number_cluster, omega_sparsity)
            obj.X = data_object.data;
            obj.K = number_cluster;
            obj.omega_sparsity = omega_sparsity;
            obj.init_method = "none";

            obj.n = size(obj.X, 2);
            obj.p = size(obj.X, 1);

            C = 0.5;
            obj.cutoff = log(1 / C) / log((1 + C) / C);


            obj.n_iter = NaN;
            obj.set_bayesian_parameters();
        end

        function fit_predict(obj, n_iter, cluster_true)
            obj.n_iter = n_iter;
            obj.initialize_cluster_est();
            obj.initialize_saving_matrix()

            for i = 1:n_iter
                variable_subset_now = obj.choose();
                disp(['Iteration ', num2str(i), ' - arms pulled: ', mat2str(find(variable_subset_now))]);
                disp(['number of arms pulled: ', mat2str(sum(variable_subset_now))]);
                reward_now = obj.reward(variable_subset_now, i);
                obj.update(variable_subset_now, reward_now);

                obj.evaluate_accuracy(cluster_true, i);
            end
            
            %final clustering
            final_selection = obj.signal_entry_est;
            X_sub_final = obj.X(final_selection, :);
            kmeans_learner = sdp_kmeans(X_sub_final, obj.K);
            obj.cluster_est_dict(obj.n_iter + 1) = cluster_est(kmeans_learner.fit_predict());
            obj.evaluate_accuracy(cluster_true, obj.n_iter + 1);
        end
        
        function evaluate_accuracy(obj, cluster_true, iter)
            cluster_est_now = obj.cluster_est_dict(iter);
            obj.acc_dict(iter) = cluster_est_now.evaluate_accuracy(cluster_true);
            obj.acc_dict(iter)
        end % end of method evaluate_accuracy

        function initialize_saving_matrix(obj)
            obj.x_tilde_est       = zeros(obj.p, obj.n, obj.n_iter);
            obj.omega_est_time    = zeros(obj.n_iter, 1);
            obj.sdp_solve_time    = zeros(obj.n_iter, 1);
            obj.entries_survived  = zeros(obj.n_iter, obj.p);
            obj.obj_val_prim      = zeros(obj.n_iter, 1);
            obj.obj_val_dual      = zeros(obj.n_iter, 1);
            obj.obj_val_original  = zeros(obj.n_iter, 1);
        end

        function cluster_est_obj = fetch_cluster_est(obj,iter)
            cluster_est_obj = obj.cluster_est_dict(iter);
        end

    end
end