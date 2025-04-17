function [cluster_est_final, iter_stop] = run_iterative_algorithm(ik, max_n_iter, window_size_half, percent_change, run_full, loop_detect_start)
%% Run_iterative_algorithm
% @export
        ik.stop_decider = stopper(max_n_iter, window_size_half, percent_change, loop_detect_start);
        ik.initialize_saving_matrix(max_n_iter)
  
        %initialization
        initial_cluster_assign = ik.get_initial_cluster_assign();
        ik.insert_cluster_est(initial_cluster_assign, 0);
        
        for iter = 1:max_n_iter
            ik.run_single_iter(iter)
            
            % stopping criterion
            criteria_vec = ik.stop_decider.apply_criteria(ik.obj_val_original, ik.obj_val_prim, iter);
            [is_stop, final_iter] = ik.stop_decider.is_stop_by_two(iter)
            if is_stop
                ik.iter_stop = final_iter;
                fprintf("\n final iteration = %i ", ik.iter_stop)
                if ~run_full
                    break 
                end
            end %end of stopping criteria
        end % end one iteration
        cluster_est_final = ik.fetch_cluster_est(ik.iter_stop);
        iter_stop = ik.iter_stop;
    end
%% 
%% 
