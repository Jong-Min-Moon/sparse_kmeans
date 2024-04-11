classdef stopperTest < matlab.unittest.TestCase
    methods(Test)
        function loop_decision_return_false(testCase)
            sp = stopper();
            actSolution = sp.compare_in_window([2,2,3,2,2], 0.3)
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of loop_decision_return_true
        
        function loop_decision_return_true(testCase)
            sp = stopper();
            actSolution = sp.compare_in_window([2,2,3,2,2], 0.5)
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of loop_decision_return_false

    end%end of methods
end%end of stopperTest