classdef data_generator_t_correlated < data_generator_t
%% data_generator_t_correlated
% @export
 
    methods
    
        function obj = data_generator_t_correlated(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_t(n, p, s, sep, seed, cluster_1_ratio);
        end
  
        function get_cov(obj)
            obj.precision = get_precision_band(obj.p, 2, 0.45);
            obj.Sigma = inv(obj.precision);
        end
     
        function [X,label] = get_data(obj, df, sd)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix();
            noise_matrix = obj.get_noise_matrix(df, sd);
            X = sqrtm(obj.Sigma) * noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
