classdef data_generator_different_cov < data_generator_t
%% data_generator_different_cov
% @export
 
    methods
    
        function obj = data_generator_different_cov(n, p, s, sep, seed, cluster_1_ratio)
            obj = obj@data_generator_t(n, p, s, sep, seed, cluster_1_ratio);
            
        end
        function noise_matrix = get_noise_matrix(obj, sd, delta)
            % Generate noise once
            rng(obj.seed);
            noise_matrix_1 = sd*(1+delta)*normrnd(0,1,[obj.p, obj.n1]);  % p x n1 noise
            noise_matrix_2 = sd*(1-delta)*normrnd(0,1,[obj.p, obj.n2]);  % p x n2 noise
            noise_matrix = [noise_matrix_1, noise_matrix_2];
            empirical_sd_1 = std(noise_matrix_1, 0, 'all');
            empirical_sd_2 = std(noise_matrix_2, 0, 'all');
            fprintf('--- empirical_sd =%f, %f  ---\\n', empirical_sd_1, empirical_sd_2);
            
        end
        function [X,label] = get_data(obj,  sd, delta)
            obj.get_cov();
            label = obj.get_cluster_label();
            mean_matrix= obj.get_mean_matrix();
            noise_matrix = obj.get_noise_matrix(sd, delta);
            X = noise_matrix + mean_matrix;
        end
    end % end of method
    
end% end of class
%% 
