
classdef block_iteration_for_server_test < matlab.unittest.TestCase
    methods(Test)
        function one_iter_test(testCase)
            bifs = block_iteration_for_server("_", "_", 1:10, 4, 100, 0.45, 500)
            database_subtable = bifs.run_one_iteration(1, 1)
            actSolution = database_subtable{9, ["acc", "obj_prim", "obj_dual", "obj_original", "discov_true", "discov_false"]}
            expSolution = [0.956, -5.8638,-5.8638, 2086.9, 10, 6 ];
            testCase.verifyEqual(actSolution, expSolution, "RelTol",1e-3)
        end

        function one_iter_test_hard(testCase)
            bifs = block_iteration_for_server("_", "_", 1:10, 3, 300, 0.45, 500)
            database_subtable = bifs.run_one_iteration(6, 4)
            actSolution = bifs.learner.iter_stop
            expSolution = 15;
            testCase.verifyEqual(actSolution, expSolution, "RelTol",1e-1)
        end
    end
end
