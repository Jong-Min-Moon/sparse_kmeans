classdef data_gaussian
    properties
        dimension
        sample_size
        data
        sample_covariance
    end

    methods
        function obj = data_gaussian(data_matrix)
            obj.data = data_matrix;
            obj.dimension = size(obj.data,1);
            obj.sample_size = size(obj.data,2);
            obj.sample_covariance = obj.data * obj.data' / (obj.sample_size-1);
        end
    end
end