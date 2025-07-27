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
        
        function noise_matrix = get_noise_matrix(obj, df, sd)
            % Generate noise once
            rng(obj.seed);
            noise_matrix = trnd(df,[obj.p, obj.n]);  % p x n noise
            sd_for_df = sqrt( df/(df-2) );
            noise_matrix = noise_matrix * sd/sd_for_df;
            noise_matrix = sqrtm(obj.Sigma) * noise_matrix;
        end
   
    end % end of method
    
end% end of class
%% 
