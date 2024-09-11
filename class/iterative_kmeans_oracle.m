classdef iterative_kmeans_oracle < iterative_kmeans


methods
    function iko = iterative_kmeans_oracle(data, number_cluster, omega_sparsity, init_method, covariance, precision)
        iko.data_object = data_gaussian_oracle(data, omega_sparsity, covariance, precision);
        iko.number_cluster = number_cluster;
        iko.omega_sparsity = omega_sparsity;
        iko.init_method = init_method;
       
    end
    
  
    %

    
end %end of methods
end %end of classdef

    

