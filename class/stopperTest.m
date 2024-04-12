classdef stopperTest < matlab.unittest.TestCase
    methods(Test)
        function apply_criteria_first(testCase)
            sp = stopper(100, 3, 0.5);
            actSolution = sp.apply_criteria([1,0,0,0], [1,0,0,0], ["","", "", ""], 1)
            expSolution = dictionary(["original", "sdp", "loop"], [false,false,false]);
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_first

        function apply_criteria_original(testCase)
            sp = stopper(100, 3, 0.5);
            actSolution = sp.apply_criteria([2,1,0.6,0], [1,10,1,0], ["sdp","loop", "", ""], 3)
            expSolution = dictionary(["original", "sdp", "loop"], [true,false,false]);
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_original

        function apply_criteria_sdp(testCase)
            sp = stopper(100, 3, 0.5);
            actSolution = sp.apply_criteria([1,10,1,0], [2,1,0.6,0],  ["original","loop", "", ""], 3)
            expSolution = dictionary(["original", "sdp", "loop"], [false, true, false]);
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_original

        function apply_criteria_loop(testCase)
            sp = stopper(100, 5, 0.00001);
            actSolution = sp.apply_criteria([1,5,1,5,1,5,0,0,0], [0,0,0,0,0,0,0,0,0],  ["","sdp", "", "", "", "","","",""], 6)
            expSolution = dictionary(["original", "sdp", "loop"], [false, false, true]);
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_loop

        function apply_criteria_nonstop(testCase)
            sp = stopper(100, 5, 0.00001);
            actSolution = sp.apply_criteria([10,9,8,7,6,5,4,3], [10,9,8,7,6,5,4,3],  ["","", "", "", "", "","","",""], 8)
            expSolution = dictionary(["original", "sdp", "loop"], [false, false, false]);
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_nonstop

        function detect_relative_change_true(testCase)
            sp = stopper(100, 3, 0.5);
            actSolution = sp.detect_relative_change( [1,1,1,0], ["","loop", "", "sdp"], 3, "sdp")
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of detect_relative_change_true

        function detect_relative_change_false(testCase)
            sp = stopper(100, 3, 0.5);
            actSolution = sp.detect_relative_change( [1,1,0.3,0], ["loop","sdp", "", "original"], 3, "original")
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of detect_relative_change_true

        function detect_loop_true(testCase)
            sp = stopper(100, 3, 0.5);
            actSolution = sp.detect_loop([2,2,2,3,2,0], [2,2,2,3,2,0], ["","","","","",""], 5)
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of detect_loop_true

        function detect_loop_false(testCase)
            sp = stopper(100, 3, 0.3);
            actSolution = sp.detect_loop([2,2,2,10,2,0], [2,2,2,10,2,0], ["","","","","",""], 5)
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of detect_loop_true

        function compare_in_window_return_false(testCase)
            sp = stopper(100, 5, 0.3);
            actSolution = sp.compare_in_window([2,2,3,2,2]);
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of loop_decision_return_true
        
        function compare_in_window_return_true(testCase)
            sp = stopper(100, 5, 0.5);
            actSolution = sp.compare_in_window([2,2,3,2,2]);
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of loop_decision_return_false
        
        function check_early_true(testCase)
            sp = stopper(100, 3, 0.01);
            actSolution = sp.check_early(1,1);
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_true

        function check_early_false(testCase)
            sp = stopper(100, 3, 0.01);
            actSolution = sp.check_early(299,5);
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_true

        function check_already_true(testCase)
            sp = stopper(100, 3, 0.01);
            actSolution = sp.check_already(["", "", "", "loop", "", "", "", "original"],5, "loop");
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_true

        function check_already_false(testCase)
            sp = stopper(100, 3, 0.01);
            actSolution = sp.check_already(["", "", "", "original", "", "", "", "loop"],5, "loop");
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_false

    end%end of methods
end%end of stopperTest