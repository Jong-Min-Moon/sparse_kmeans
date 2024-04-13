classdef cluster_est_test < matlab.unittest.TestCase
    methods(Test)
        function create_partition_dict(testCase)
            ce = cluster_est([1,2,1,2]);
            actSolution = ce.cluster_partition;
            expSolution = dictionary([1,2], {[1,3], [2,4]});
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_first


    end%end of methods
end%end of stopperTest