classdef sdp_kmeans_bnb < handle
    properties
        X           % Data matrix (p x n)
        K           % Number of clusters
        s           % Sparsity level (if s-constraint used)
        lambda      % Penalty parameter (if penalty used)
        n           % Number of data points
        p           % Data dimension
        
        % B&B State
        best_obj    % Global Lower Bound (Best Integer Objective)
        best_sol    % Structure with {Z, w, features}
        
        % Options
        use_penalty % Boolean: true for penalty, false for s-constraint
        max_nodes   % Safety limit
        tol         % Numerical tolerance
        
        % Solver Handles (from literate_programming.m)
        solver_func 
        converter_func
    end
    
    methods
        function obj = sdp_kmeans_bnb(X, K, s_or_lambda, use_penalty)
            obj.X = X;
            obj.K = K;
            obj.n = size(X, 2);
            obj.p = size(X, 1);
            obj.use_penalty = use_penalty;
            
            if use_penalty
                obj.lambda = s_or_lambda;
                obj.s = inf;
            else
                obj.s = s_or_lambda;
                obj.lambda = 0;
            end
            
            obj.best_obj = -inf;
            obj.max_nodes = 1000;
            obj.tol = 1e-4;
            
            % Assume these are available in path
            obj.solver_func = @kmeans_sdp_pengwei; 
            obj.converter_func = @sdp_sol_to_cluster;
        end
        
        function [final_cluster, final_features, solve_log, initial_sol] = solve(obj)
            % 1. Initialization (Using Iterative SDP with Statistical Cutoff)
            fprintf('--- Phase 1: Statistical Initialization (No s required) ---\n');
            
            iter_solver = sdp_kmeans_iter_knowncov(obj.X, obj.K);
            [~, labels_init] = iter_solver.fit_predict(10, 5, 3, 1e-3);
            
            % Derive Lambda if not set (using the paper's threshold logic)
            n1 = sum(labels_init == 1); n2 = obj.n - n1;
            if n1 == 0 || n2 == 0
                warning('Initial Clustering failed. Falling back to default lambda.');
                if isempty(obj.lambda) || obj.lambda == 0
                    obj.lambda = 2 * log(obj.p); % Default heuristic
                end
            else
                % Use the statistical cutoff derived from the cluster sizes
                % Target: abs_diff > sqrt( 2 * log(p) * n / (n1*n2) )
                % In Obj (score) space, this is square of that.
                % Use the statistical cutoff (2 * log(p)) as the penalty for T_j^2
                if isempty(obj.lambda) || obj.lambda == 0
                    obj.lambda = 2 * log(obj.p);
                    fprintf('Determined Statistical Lambda: %.4f\n', obj.lambda);
                end
            end
            
            % 2. Improved Warm Start
            fprintf('--- Phase 2: Refined Warm Start (Iterative) ---\n');
            [Z_heur, w_heur, obj_heur] = obj.run_heuristic();
            
            % Screening based on Penalty + Refined Scores
            scores_ref = obj.get_scores(Z_heur); 
            screen_mask = scores_ref > (obj.lambda * 0.5);
            % Safety: never screen out features already selected by the heuristic
            screen_mask(w_heur == 1) = true; 
            
            obj.best_obj = obj_heur;
            obj.best_sol = struct('Z', Z_heur, 'w', w_heur);
            
            initial_sol.obj = obj.best_obj;
            initial_sol.w = w_heur;
            initial_sol.Z = Z_heur;
            
            fixed_vars = -ones(obj.p, 1);
            fixed_vars(~screen_mask) = 0;
            
            fprintf('Initial LB (Heuristic): %.4f, Features: %d\n', obj_heur, sum(w_heur));
            fprintf('Screened out %d features. Remaining: %d\n', sum(~screen_mask), sum(screen_mask));
            
            % 3. Branch and Bound
            fprintf('--- Phase 3: Branch and Bound (Penalty Mode) ---\n');
            queue = {};
            root = struct('fixed', fixed_vars, 'depth', 0, 'id', 1);
            queue{end+1} = root;
            
            nodes_processed = 0;
            while ~isempty(queue) && nodes_processed < obj.max_nodes
                nodes_processed = nodes_processed + 1;
                node = queue{end}; queue(end) = [];
                
                [res, is_feasible] = obj.solve_relaxation(node.fixed);
                if ~is_feasible || (res.ub <= obj.best_obj + obj.tol), continue; end
                
                if res.ub_integer > obj.best_obj
                    obj.best_obj = res.ub_integer;
                    obj.best_sol = res.sol_integer;
                    fprintf('Node %d: New Best! Obj: %.4f, s: %d\n', node.id, obj.best_obj, sum(obj.best_sol.w));
                end
                
                free_indices = find(node.fixed == -1);
                if isempty(free_indices), continue; end
                
                % Branching
                beta_node = obj.get_discriminating_direction(res.Z);
                [~, sort_idx] = sort(abs(beta_node(free_indices)), 'descend');
                branch_idx = free_indices(sort_idx(1));
                
                c2 = node; c2.fixed(branch_idx) = 0; c2.id = nodes_processed*2 + 1;
                c1 = node; c1.fixed(branch_idx) = 1; c1.id = nodes_processed*2;
                
                queue{end+1} = c2; queue{end+1} = c1;
            end
            


            

            
            % Final Result
            final_cluster = obj.converter_func(obj.best_sol.Z, obj.K);
            final_features = find(obj.best_sol.w);
            solve_log.obj = obj.best_obj;
            solve_log.nodes = nodes_processed;
        end
        
        function [res, is_feasible] = solve_relaxation(obj, fixed_vec)
            % fixed_vec: -1 (Free), 0 (Fixed-0), 1 (Fixed-1)
            
            % 1. Identify active pool
            fixed_1 = find(fixed_vec == 1);
            free    = find(fixed_vec == -1);
            
            if obj.use_penalty
                subset = [fixed_1; free];
            else
                % S-coord constraint
                if length(fixed_1) > obj.s
                    res = []; is_feasible = false; return;
                end
                subset = [fixed_1; free]; 
            end
            
            if isempty(subset)
                 res = []; is_feasible = true; 
                 res.ub = -inf; res.ub_integer = -inf;
                 return;
            end
            
            % Construct Affinity Matrix for this subset
            X_sub = obj.X(subset, :);
            D = X_sub' * X_sub; 
            
            % Solve SDP
            Z = obj.solver_func(D, obj.K);
            
            % Compute Indvidual Contributions (Scores)
            scores = zeros(obj.p, 1);
            XZ = X_sub * Z;
            scores(subset) = sum(XZ .* X_sub, 2); % Row-wise dot product
            
            % --- Compute Upper Bound ---
            ub = 0;
            
            if obj.use_penalty
                % UB = Sum_{j \in Fixed1} (score_j - lambda) 
                %    + Sum_{j \in Free} max(0, score_j - lambda)
                ub = sum(scores(fixed_1) - obj.lambda);
                ub = ub + sum(max(0, scores(free) - obj.lambda));
                
                % Construct Heuristic Integer Solution
                w_int = zeros(obj.p, 1);
                w_int(fixed_1) = 1;
                w_int(free(scores(free) > obj.lambda)) = 1;
                
            else
                % Constraint sum(w) <= s
                ub = sum(scores(fixed_1));
                rem_s = obj.s - length(fixed_1);
                if rem_s > 0 && ~isempty(free)
                    [sorted_vals, ~] = sort(scores(free), 'descend');
                    take = min(length(sorted_vals), rem_s);
                    ub = ub + sum(sorted_vals(1:take));
                end
                
                % Construct Heuristic Integer Solution
                w_int = zeros(obj.p, 1);
                w_int(fixed_1) = 1;
                if rem_s > 0 && ~isempty(free)
                   [~, sort_idx] = sort(scores(free), 'descend');
                    take = min(length(sort_idx), rem_s);
                    w_int(free(sort_idx(1:take))) = 1;
                end
            end
            
            ub_integer = obj.evaluate_integer(w_int);
            
            res.ub = ub;
            res.ub_integer = ub_integer;
            res.Z = Z;
            res.sol_integer = struct('Z', Z, 'w', w_int); % Approx Z
            is_feasible = true;
        end
        
        function obj_val = evaluate_integer(self, w)
            subset = find(w);
            if isempty(subset)
                obj_val = -inf; return;
            end
            X_sub = self.X(subset, :);
            D = X_sub' * X_sub;
            Z = self.solver_func(D, self.K);
            
            % True objective
            scores = sum((X_sub * Z) .* X_sub, 2);
            raw_sum = sum(scores);
            
            if self.use_penalty
                obj_val = raw_sum - self.lambda * sum(w);
            else
                obj_val = raw_sum;
            end
        end
        
        function [Z, w, obj_val] = run_heuristic(self)
            % Implement Algorithm 2 (Iterative SDP K-means)
            % loops T times or until convergence
            
            max_iter = 5;
            
            % 1. Initialization: Use All Features (Standard Algorithm 2)
            % This matches the deterministic initialization.
            current_features = 1:self.p;
            
            w = zeros(self.p, 1);
            w(current_features) = 1;
            
            last_features = [];
            
            for t = 1:max_iter
                % Step 1: Solve SDP on current subset
                if isempty(current_features)
                     % Should not happen, but safe fallback
                     current_features = randperm(self.p, self.s);
                end
                
                X_sub = self.X(current_features, :);
                D = X_sub' * X_sub;
                Z = self.solver_func(D, self.K);
                
                % Step 2: Feature Selection based on Z
                scores = self.get_scores(Z);
                [~, idx] = sort(scores, 'descend');
                
                w = zeros(self.p, 1);
                if self.use_penalty
                     w(scores > self.lambda) = 1;
                else
                     w(idx(1:self.s)) = 1;
                end
                
                if sum(w) == 0 && self.use_penalty
                    % Safety: select at least the best one
                    [~, best_idx] = max(scores);
                    w(best_idx) = 1;
                end
                
                current_features = find(w);
                
                % Check Convergence
                if isequal(sort(current_features(:)), sort(last_features(:)))
                    break;
                end
                last_features = current_features;
            end
            
            % Final Evaluation
            obj_val = self.evaluate_integer(w);
            
            % Ensure Z matches final w
            X_final = self.X(current_features, :);
            D_final = X_final' * X_final;
            Z = self.solver_func(D_final, self.K);
        end
        
        function scores = get_scores(self, Z)
            labels = self.converter_func(Z, self.K);
            n1 = sum(labels == 1);
            n2 = self.n - n1;
            if n1 == 0 || n2 == 0
                scores = zeros(self.p, 1);
                return;
            end
            m1 = mean(self.X(:, labels == 1), 2);
            m2 = mean(self.X(:, labels == 2), 2);
            scores = (m1 - m2).^2 * (n1 * n2 / self.n);
        end
        
        function beta = get_discriminating_direction(self, Z)
             labels = self.converter_func(Z, self.K);
             m1 = mean(self.X(:, labels == 1), 2);
             m2 = mean(self.X(:, labels == 2), 2);
             beta = m1 - m2;
        end
    end
end
