classdef data_generator_subsample < handle
%% data_generator_subsample
% @export
    properties
        X        % Data matrix (d x n)
        y
        n           % Number of data points
     percent_cluster_1
 subsample_size_cluster_1
 subsample_size_cluster_2
  
    end
    methods
    
        function obj = data_generator_subsample(X, y)
            obj.n = size(X, 2);
            obj.y = y;
             obj.X  =X;
             obj.percent_cluster_1 = sum(y==1)/sum(y>0);
        end
 
        function [X_new,y_new] = get_data(obj, subsample_size, seed)
            rng(seed);
            idx_cluster_1 = find(obj.y == 1);
            idx_cluster_2 = find(obj.y == 2); % Assuming cluster 2 is the other cluster
            
                         obj.subsample_size_cluster_1 = floor(subsample_size * obj.percent_cluster_1);
            obj.subsample_size_cluster_2 = subsample_size - obj.subsample_size_cluster_1;
            
            %pseudocode
            perm_idx_cluster_1 = randperm(numel(idx_cluster_1));
            selected_idx_cluster_1 = idx_cluster_1(perm_idx_cluster_1(1:obj.subsample_size_cluster_1));
            
            % --- Select samples from cluster 2 ---
            perm_idx_cluster_2 = randperm(numel(idx_cluster_2));
            selected_idx_cluster_2 = idx_cluster_2(perm_idx_cluster_2(1:obj.subsample_size_cluster_2));
            final_idx = [selected_idx_cluster_1, selected_idx_cluster_2];
            
            X_new =  obj.X(:,final_idx);
            y_new = obj.y(final_idx);
                 
        end
    end % end of method
    
end% end of class
