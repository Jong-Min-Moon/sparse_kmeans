classdef stopperTest < matlab.unittest.TestCase
    methods(Test)
        function compare_in_window_return_false(testCase)
            sp = stopper(0.3);
            actSolution = sp.compare_in_window([2,2,3,2,2]);
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of loop_decision_return_true
        
        function compare_in_window_return_true(testCase)
            sp = stopper(0.5);
            actSolution = sp.compare_in_window([2,2,3,2,2]);
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of loop_decision_return_false
        
        function check_early_true(testCase)
            sp = stopper(0.01);
            actSolution = sp.check_early(1,1);
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_true

        function check_early_false(testCase)
            sp = stopper(0.01);
            actSolution = sp.check_early(299,5);
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_true

        function check_already_true(testCase)
            sp = stopper(0.01);
            actSolution = sp.check_already(["", "", "", "loop", "", "", "", "original"],5, "loop");
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_true

        function check_already_false(testCase)
            sp = stopper(0.01);
            actSolution = sp.check_already(["", "", "", "original", "", "", "", "loop"],5, "loop");
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_false

    end%end of methods
end%end of stopperTest