classdef stopper < handle
    properties
        window_size
        percent_change
        max_iter
        stop_history
        is_stop
        final_iter_calculation
        final_iter_return
        loop_detect_start
    end
    methods
        function sp = stopper(max_iter, window_size_half, percent_change, loop_detect_start)
            sp.window_size = 1 + window_size_half*2;
            sp.percent_change = percent_change;
            sp.max_iter = max_iter;
            sp.is_stop = false;
            sp.loop_detect_start = loop_detect_start;
            original = repelem(false, max_iter, 1);
            sdp = repelem(false, max_iter, 1);
            loop = repelem(false, max_iter, 1);
            sp.stop_history = table(original, sdp, loop);
        end



        function [is_stop, final_iter_return]  = is_stop_by_two(sp,iter)
            criterion_vec = ["original", "sdp", "loop"];
            is_converge = arrayfun(@(criterion) ~isempty(find(sp.stop_history{1:iter,criterion})), criterion_vec);
            criteron_activated = criterion_vec(is_converge);
            index_converge = arrayfun(@(criterion) find(sp.stop_history{1:iter,criterion}), criteron_activated);

            is_stop = ( sum(is_converge) >= 2) | (iter == sp.max_iter);
            sp.is_stop = is_stop;
            if is_stop
                sp.final_iter_calculation = iter;
                if iter < sp.max_iter
                    sp.final_iter_return = max(index_converge);
                elseif iter == sp.max_iter
                    sp.final_iter_return = sp.max_iter;
                end
            else
                sp.final_iter_return = 1;
            end
            final_iter_return = sp.final_iter_return;
        end
        
        function criteria_vec = apply_criteria(sp, obj_val_original, obj_val_sdp, iter)
            criteria_vec = sp.stop_history(iter,:);
            if iter>1
                is_converge_original = sp.detect_relative_change(obj_val_original, iter, "original");
                criteria_vec{1, "original"} = is_converge_original;
                if is_converge_original
                    sp.stop_history{iter, "original"} = is_converge_original;
                end

                is_converge_sdp = sp.detect_relative_change(obj_val_sdp, iter, "sdp");
                criteria_vec{1, "sdp"} = is_converge_sdp;
                if is_converge_sdp
                    sp.stop_history{iter, "sdp"} = is_converge_sdp;
                end

                is_loop = sp.detect_loop(obj_val_original, obj_val_sdp, iter);
                criteria_vec{1, "loop"} = is_loop;
                if is_loop
                    sp.stop_history{iter - (sp.window_size-1)/2, "loop"} = is_loop;
                end
            end
        end

        function is_converge = detect_relative_change(sp, obj_val_vec, iter, criteria)
            if (sp.check_already(iter, criteria) | (get_relative_change(obj_val_vec, iter) > sp.percent_change))
                is_converge = false;
            else
                is_converge = true;
            end                    
        end %end of detect_relative_change



        function is_loop = detect_loop(sp, obj_val_original, obj_val_sdp, iter)
            is_early = iter <= sp.loop_detect_start;
            if (is_early | sp.check_already(iter, "loop"))
                is_loop = false;
            else
                window_vec_original = obj_val_original(iter-(sp.window_size-1):iter );
                window_vec_sdp      = obj_val_sdp(     iter-(sp.window_size-1):iter );
                is_loop = ( sp.compare_in_window(window_vec_sdp) | sp.compare_in_window(window_vec_original) );  
            end
        end

        function is_loop = detect_loop(sp, obj_val_original, obj_val_sdp, iter)
            is_early = iter <= sp.loop_detect_start;
            if (is_early | sp.check_already(iter, "loop"))
                is_loop = false;
            else
                window_vec_original = obj_val_original(iter-(sp.window_size-1):iter );
                window_vec_sdp      = obj_val_sdp(     iter-(sp.window_size-1):iter );
                is_loop = ( sp.compare_in_window(window_vec_sdp) | sp.compare_in_window(window_vec_original) );  
            end
        end
        function decision_loop = compare_in_window(sp, window_vec)
            if length(window_vec) == sp.window_size
                index_center = (sp.window_size+1)/2;
                value_center = window_vec(index_center);
                value_window = window_vec;
                change = abs(value_window - value_center)/value_center
                if sum( change < sp.percent_change) >1
                    decision_loop = true;
                else
                    decision_loop = false;
                end
            else
                error("the window size does not match")
            end
        end %end of method loop_decision
        
        function is_already = check_already(sp, iter, criteria) 
            is_already =  sum(sp.stop_history{1:(iter-1), criteria}) > 0;
        end%end of is_already



    end%end of methods
end%end of classdef