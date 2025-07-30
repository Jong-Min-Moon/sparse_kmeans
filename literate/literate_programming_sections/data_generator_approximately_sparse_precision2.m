classdef data_generator_approximately_sparse_precision2 < data_generator_approximately_sparse_precision
%% data_generator_approximately_sparse_precision2
% @export
    methods
        function get_cov(obj, delta)
           obj.precision = get_precision_band(obj.p, 2, 0.45);
           obj.precision(obj.precision == 0) = delta;     
           obj.Sigma = inv(obj.precision);
        end
    
 
 
    end % end of methods
end
 
