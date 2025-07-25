classdef data_generator_t < handle
%% data_generator_t
% @export
    properties
        X           % Data matrix (d x n)
        y           % cluster label
        K           % Number of clusters
        n           % Number of data points
        n1
        n2
        sep
        seed
        s
        p           % Data dimensions
        cutoff      % Threshold for variable inclusion
        n_iter
        Sigma
        precision
 
  
    end
    methods
    
        function obj = data_generator_t(n, p, s, sep, seed, cluster_1_ratio)
            obj.n = n;
            obj.p = p;
            obj.s = s;
            obj.sep = sep;
            obj.seed = seed;
            obj.n1 = round(n * cluster_1_ratio);
            obj.n2 = n - obj.n1;
            
        end
        function label = get_cluster_label(obj)
            label = [ones(obj.n1, 1); 2 * ones(obj.n2, 1)];
            label = label';
        end
        function get_cov(obj)
            obj.Sigma = eye(obj.p);
            obj.precision = obj.Sigma;
        end
        function beta_star = get_beta(obj)
             beta_star = zeros(obj.p, 1);
             beta_star(1:obj.s) = 1;
             M= (obj.sep)/2/ sqrt( sum( obj.Sigma(1:obj.s,1:obj.s),"all") );
             beta_star = M * beta_star;
        end
        function mean_matrix = get_mean_matrix(obj)
             beta  = obj.get_beta();
                    % Set class means
             mu1 = obj.precision \ beta ;
             mu2 = -mu1;
             % Create mean matrix
             mean_matrix = [repmat(mu1', obj.n1, 1); repmat(mu2', obj.n2, 1)];
             mean_matrix= mean_matrix';
        end
        function noise_matrix = get_noise_matrix(obj, df, sd)
            % Generate noise once
            rng(obj.seed);
            noise_matrix = trnd(df,[obj.p, obj.n]);  % n x p noise
            sd_for_df = sqrt( df/(df-2) );
            noise_matrix = noise_matrix * sd/sd_for_df;
            empirical_sd = std(noise_matrix, 0, 'all');
            fprintf('--- empirical_sd =%f  ---\\n', empirical_sd);
            
        end
        function [X,label] = get_data(obj, df, sd)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix();
            noise_matrix = obj.get_noise_matrix(df, sd);
            X = noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
%% 
