classdef variance_test_spectral_arias < variance_test_spectral_azyzan


    methods
   


        function threshold = get_threshold (obj)
            threshold = 1 + 5 * (sqrt(log(obj.p) / obj.n) + log(obj.p) / obj.n)
        end


   
    end % end of methods
end
