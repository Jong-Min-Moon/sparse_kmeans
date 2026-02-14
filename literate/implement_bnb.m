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
        
        function [final_cluster, final_features, log] = solve(obj)
            % 1. Initialization (Heuristic Warm Start)
            fprintf('--- Phase 1: Warm Start (Iterative Heuristic) ---\n');
            [Z_heur, w_heur, obj_heur] = obj.run_heuristic();
            
            obj.best_obj = obj_heur;
            obj.best_sol = struct('Z', Z_heur, 'w', w_heur);
            fprintf('Initial LB: %.4f, Features: %d\n', obj_heur, sum(w_heur));
            
            % 2. Screening
            fprintf('--- Phase 2: Screening ---\n');
            beta_est = obj.get_discriminating_direction(Z_heur);
            % Keep features with top 2*s or cutoff (simple screening)
            % For penalty: keep if beta > small_epsilon
            screen_mask = abs(beta_est) > 1e-4 * max(abs(beta_est)); 
            if ~obj.use_penalty
                 [~, idx] = sort(abs(beta_est), 'descend');
                 keep_n = min(obj.p, obj.s * 3); % Conservative screening
                 screen_mask = false(obj.p, 1);
                 screen_mask(idx(1:keep_n)) = true;
            end
            
            fixed_vars = -ones(obj.p, 1); % -1: Free
            fixed_vars(~screen_mask) = 0; % Fix to 0
            
            fprintf('Screened out %d features. Remaining: %d\n', sum(~screen_mask), sum(screen_mask));
            
            % 3. Branch and Bound
            fprintf('--- Phase 3: Branch and Bound ---\n');
            queue = {};
            % Root Node
            root = struct('fixed', fixed_vars, 'depth', 0, 'id', 1);
            queue{end+1} = root;
            
            nodes_processed = 0;
            
            while ~isempty(queue) && nodes_processed < obj.max_nodes
                nodes_processed = nodes_processed + 1;
                
                % Strategy: Depth First (pop end)
                node = queue{end};
                queue(end) = []; 
                
                % Solve Relaxation
                [res, is_feasible] = obj.solve_relaxation(node.fixed);
                
                if ~is_feasible
                     fprintf('Node %d: Infeasible via constraints.\n', node.id);
                     continue;
                end
                
                % Pruning
                if res.ub <= obj.best_obj + obj.tol
                    % Prune by Bound
                    continue;
                end
                
                % Check Integer Feasibility
                % In our relaxation, w is implicitly "soft" because we compute UB by picking top features
                % We need to check if we can make a VALID integer solution from this
                % Using the heuristic integer construction from relaxation:
                if res.ub_integer > obj.best_obj
                    obj.best_obj = res.ub_integer;
                    obj.best_sol = res.sol_integer;
                    fprintf('Node %d: New Best Found! Obj: %.4f\n', node.id, obj.best_obj);
                end
                
                % Branching
                % Find best free variable to branch on
                free_indices = find(node.fixed == -1);
                if isempty(free_indices)
                    continue; % Leaf
                end
                
                % Branching Rule: Max discriminating direction
                beta_node = obj.get_discriminating_direction(res.Z);
                [~, sort_idx] = sort(abs(beta_node(free_indices)), 'descend');
                branch_idx = free_indices(sort_idx(1));
                
                % Create Children
                % Child 1: w_j = 1 (Promising)
                child1 = node;
                child1.fixed(branch_idx) = 1;
                child1.depth = node.depth + 1;
                child1.id = nodes_processed * 2;
                
                % Child 2: w_j = 0
                child2 = node;
                child2.fixed(branch_idx) = 0;
                child2.depth = node.depth + 1;
                child2.id = nodes_processed * 2 + 1;
                
                % Push promising first (for DFS)
                queue{end+1} = child2;
                queue{end+1} = child1;
            end
            
            % Final Result
            final_cluster = obj.converter_func(obj.best_sol.Z, obj.K);
            final_features = find(obj.best_sol.w);
            log.obj = obj.best_obj;
            log.nodes = nodes_processed;
        end
        
        function [res, is_feasible] = solve_relaxation(obj, fixed_vec)
            % fixed_vec: -1 (Free), 0 (Fixed-0), 1 (Fixed-1)
            
            % 1. Identify active pool
            % To compute a valid Upper Bound, we relax the problem.
            % Relaxation: Allowed to pick any 'Free' variables up to s (or penalty)
            % Strategies to get Upper Bound:
            % Run SDP on (Fixed-1 + Free) features. 
            % Then retrospectively select the best ones from Free.
            
            fixed_1 = find(fixed_vec == 1);
            free    = find(fixed_vec == -1);
            
            if obj.use_penalty
                % Penalty Case: UB is simple. 
                % Solve SDP with ALL (Fixed-1 + Free).
                % Obj_contribution(j) - lambda.
                % If linear gain > lambda, keep it.
                subset = [fixed_1; free];
            else
                % S-coord constraint
                % Must have Fixed-1 <= s
                if length(fixed_1) > obj.s
                    res = []; is_feasible = false; return;
                end
                subset = [fixed_1; free]; % Use all potential
            end
            
            if isempty(subset)
                 res = []; is_feasible = true; 
                 res.ub = -inf; res.ub_integer = -inf;
                 return;
            end
            
            % Construct Affinity Matrix for this subset
            X_sub = obj.X(subset, :);
            % Normalize X_sub for numerical stability in SDP (optional, based on paper)
            % The paper uses X' * Omega * X. Assuming Omega=I for now (Known Covariance)
            % A = X^T X = sum K_j
            
            % Wait, sdp_kmeans uses normalize(X'). Let's stick to raw construction for logic
            D = X_sub' * X_sub; 
            
            % Solve SDP
            Z = obj.solver_func(D, obj.K);
            
            % Compute Indvidual Contributions (Scores)
            % score_j = <K_j, Z> = <x_j^T x_j, Z> = x_j * Z * x_j' 
            % Efficiently: diag(X * Z * X')
            % Let's compute vector of scores for ALL p (0 for fixed-0)
            scores = zeros(obj.p, 1);
            % Only compute for subset used
            % score_j = (x_j * Z) * x_j' = sum( (x_j*Z) .* x_j )
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
                % Greedily pick free > lambda
                w_int(free(scores(free) > obj.lambda)) = 1;
                
            else
                % Constraint sum(w) <= s
                % UB = Sum_{j \in Fixed1} score_j
                %    + Sum of Top (s - |Fixed1|) scores from Free
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
            % Simple mock of Algorithm 2
            % 1. Initial Cluster
            subset = 1:self.p; 
            % Random subset if p large? No, use all for simple start
            w = zeros(self.p, 1);
            
            % Initial SDP
             X_sub = self.X;
             D = X_sub' * X_sub;
             Z = self.solver_func(D, self.K);
             
             % One step update
             beta = self.get_discriminating_direction(Z);
             [~, idx] = sort(abs(beta), 'descend');
             
             if self.use_penalty
                 cutoff = self.lambda; % Rough
                 w(abs(beta) > 0) = 1; % Mock
                 % Better: use same logic as node relaxation
                 selected = 1:min(self.p, 10); % Dummy for demo
                 w(selected) = 1;
             else
                 w(idx(1:self.s)) = 1;
             end
             
             obj_val = self.evaluate_integer(w);
        end
        
        function beta = get_discriminating_direction(self, Z)
            % Estimate beta from Z
            % beta_j approx mean diff
            % For SDP, center separation is related to eigenvectors
            % Let's use the simple variance contribution as proxy for 'beta magnitude'
            % score_j = <K_j, Z>
            % Since maximizing score is the goal, branching on high score is good
            
            % Compute scores again
            % score_j = (x_j * Z) * x_j'
            % This is exactly 'Between Cluster Variance' of feature j
             XZ = self.X * Z;
             scores = sum(XZ .* self.X, 2);
             beta = scores; % Use score as proxy for discrimination
        end
    end
end
