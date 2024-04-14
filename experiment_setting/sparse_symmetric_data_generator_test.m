classdef sparse_symmetric_data_generator_test < matlab.unittest.TestCase
    methods(Test)
        function test_magnitude(testCase)
            sdg = sparse_symmetric_data_generator(1:10, 4, 100, 2, 0.45);
            [mean_1, mean_2] = sdg.get_symmetric_mean_sparse_before_innovation()
            dist_malhal_sqrd = (mean_1-mean_2)' * sdg.sparse_precision_matrix * (mean_1-mean_2);
            actSolution = sqrt( dist_malhal_sqrd );
            expSolution = 4;
            testCase.verifyEqual(actSolution, expSolution)
        end% end of evaluate_accuracy
    end%end of methods
end%end of stopperTest