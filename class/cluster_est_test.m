classdef cluster_est_test < matlab.unittest.TestCase
    methods(Test)
        function create_partition_dict(testCase)
            ce = cluster_est([1,2,1,2]);
            actSolution = ce.cluster_partition;
            expSolution = dictionary([1,2], {[1,3], [2,4]});
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_first

        function change_label(testCase)
            ce = cluster_est([1,2,1,2]); 
            actSolution = ce.change_label([2,1]);
            expSolution = [2,1,2,1];
            testCase.verifyEqual(actSolution,expSolution)
        end%change_label

        function evaluate_accuracy(testCase)
            ce = cluster_est([1,2,1,2]); 
            actSolution = ce.evaluate_accuracy([1,2,2,2]);
            expSolution = 0.75;
            testCase.verifyEqual(actSolution,expSolution)
        end% end of evaluate_accuracy
    end%end of methods
end%end of stopperTest