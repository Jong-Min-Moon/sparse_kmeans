classdef stopperTest < matlab.unittest.TestCase
    methods(Test)
        function compare_in_window_return_false(testCase)
            sp = stopper();
            actSolution = sp.compare_in_window([2,2,3,2,2], 0.3);
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of loop_decision_return_true
        
        function compare_in_window_return_true(testCase)
            sp = stopper();
            actSolution = sp.compare_in_window([2,2,3,2,2], 0.5);
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of loop_decision_return_false
        
        function check_early_true(testCase)
            sp = stopper();
            actSolution = sp.check_early(1,1);
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_true

        function check_early_false(testCase)
            sp = stopper();
            actSolution = sp.check_early(299,5);
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_true
    end%end of methods
end%end of stopperTest