classdef stopperTest < matlab.unittest.TestCase
    methods(Test)
        function apply_criteria_first(testCase)
            sp = stopper(100, 3, 0.5);
            actSolution = sp.apply_criteria([1,0,0,0], [1,0,0,0], 1)
            original = false;
            sdp = false;
            loop = false;
            expSolution = table(original, sdp, loop);
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_first

        function apply_criteria_original(testCase)
            sp = stopper(100, 5, 0.01);
            actSolution = sp.apply_criteria([2, 1, 1.9, 0.8, 0.7999], [1,10,100,1000, 10000], 5)
            original = true;
            sdp = false;
            loop = false;
            expSolution = table(original, sdp, loop);
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_original_loop

        function apply_criteria_sdp(testCase)
            sp = stopper(100, 5, 0.01);
            actSolution = sp.apply_criteria([1,10,100,1000, 10000], [2, 1, 1.9, 0.8, 0.7999], 5)
            original = false;
            sdp = true;
            loop = false;
            expSolution = table(original, sdp, loop);
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_original

        function apply_criteria_original_loop(testCase)
            sp = stopper(100, 5, 0.01);
            actSolution = sp.apply_criteria([1, 10, 100, 1000, 999.9999], [1,3,1,3,1], 5)
            original = true;
            sdp = false;
            loop = true;
            expSolution = table(original, sdp, loop);
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_original_loop

        function apply_criteria_loop(testCase)
            sp = stopper(100, 5, 0.00001);
            actSolution = sp.apply_criteria( ...
                [666.41, 1426.2, 1023.6, 1379.2,   1726, 1232.1, 1789.1, 1831.4, 1898.8, 1939.1, 1565.9, 1643.4, 1491.2, 1791.3, 1657.2, 1856.9, 1569.4, 1936.6, 1647.5, 1822.3, 1656.1, 1871.6], ...
                [-6.6188, -5.4111,  -5.713, -6.4386, -5.2692, -5.9411, -6.0581,  -6.573, -5.8132,  -6.075,  -6.481, -6.3241, -6.6013, -6.3012, -6.3302,  -5.605, -6.4672,   -6.36, -6.9014, -6.1495, -6.3179, -6.2851, -6.5507], ...
                )
            original = false;
            sdp = false;
            loop = true;
            expSolution = table(original, sdp, loop);
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_loop

        function apply_criteria_nonstop(testCase)
            sp = stopper(100, 5, 0.00001);
            actSolution = sp.apply_criteria([10,9,8,7,6,5,4,3], [10,9,8,7,6,5,4,3], 8)
            original = false;
            sdp = false;
            loop = false;
            expSolution = table(original, sdp, loop);
            testCase.verifyEqual(actSolution,expSolution)
        end%apply_criteria_nonstop

        function detect_relative_change_true(testCase)
            sp = stopper(100, 3, 0.5);
            sp.stop_history{1:4, "original"} = [false, false, false, true]';
            actSolution = sp.detect_relative_change( [1,1,1,0], 3, "sdp")
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of detect_relative_change_true

        function detect_relative_change_false(testCase)
            sp = stopper(100, 3, 0.5);
            sp.stop_history{1:4, "original"} = [false, false, false, true]';
            actSolution = sp.detect_relative_change( [1,1,0.3,0], 3, "original")
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of detect_relative_change_true

        function detect_relative_change_already(testCase)
            sp = stopper(100, 3, 0.5);
            sp.stop_history{1:4, "original"} = [false, false, true, true]';
            actSolution = sp.detect_relative_change( [1,1,1,0], 3, "original")
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of detect_relative_change_already

        function detect_loop_true(testCase)
            sp = stopper(100, 3, 0.5);
            actSolution = sp.detect_loop([2,2,2,3,2,0], [2,2,2,3,2,0], 5);
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of detect_loop_true

        function detect_loop_false(testCase)
            sp = stopper(100, 3, 0.3);
            actSolution = sp.detect_loop([2,2,2,10,2,0], [2,2,2,10,2,0], 5);
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
            sp.stop_history{1:8, "loop"} = [false, false, false, true, false, false, false, false]';
            actSolution = sp.check_already(5, "loop");
            expSolution = true;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_true

        function check_already_false(testCase)
            sp = stopper(100, 3, 0.01);
            sp.stop_history{1:8, "original"} = [false, false, false, false, false, false, true, false]';
            actSolution = sp.check_already(5, "original");
            expSolution = false;
            testCase.verifyEqual(actSolution,expSolution)
        end%end of check_early_false

    end%end of methods
end%end of stopperTest