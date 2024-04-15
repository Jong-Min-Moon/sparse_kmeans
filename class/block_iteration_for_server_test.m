
classdef block_iteration_for_server_test < matlab.unittest.TestCase
    methods(Test)
        function one_iter_test(testCase)
            bifs = block_iteration_for_server("_", "_", 1:10, 4, 100, 0.45, 500, @sparse_symmetric_data_generator)
            database_subtable = bifs.run_one_iteration(1, 1)
            actSolution = database_subtable{9, ["acc", "obj_prim", "obj_dual", "obj_original", "discov_true", "discov_false"]}
            expSolution = [0.956, -5.8638,-5.8638, 2086.9, 10, 6 ];
            testCase.verifyEqual(actSolution, expSolution, "RelTol",1e-3)
        end
    end
end
