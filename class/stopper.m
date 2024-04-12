classdef stopper < handle
    properties
        window_size
        percent_change
        max_iter
        stop_history
    end
    methods
        function sp = stopper(max_iter, window_size, percent_change)
            sp.window_size = window_size;
            sp.percent_change = percent_change;
            sp.max_iter = max_iter;

            original = repelem(false, max_iter, 1);
            sdp = repelem(false, max_iter, 1);
            loop = repelem(false, max_iter, 1);
            sp.stop_history = table(original, sdp, loop);
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
            if (sp.check_already(iter, criteria) | (sp.get_relative_change(obj_val_vec, iter) > sp.percent_change))
                is_converge = false;
            else
                is_converge = true;
            end                    
        end %end of detect_relative_change
        function relative_change = get_relative_change(sp, obj_val_vec, iter)
            relative_change = abs((obj_val_vec(iter) - obj_val_vec(iter-1))/obj_val_vec(iter-1));
        end

        function is_loop = detect_loop(sp, obj_val_original, obj_val_sdp, iter)
            if (sp.check_early(iter, sp.window_size) | sp.check_already(iter, "loop"))
                is_loop = false;
            else
                window_vec_original = obj_val_original(iter-(sp.window_size-1):iter )
                window_vec_sdp      = obj_val_sdp(     iter-(sp.window_size-1):iter )
                is_loop = ( sp.compare_in_window(window_vec_sdp) | sp.compare_in_window(window_vec_original) );  
            end
        end

        function decision_loop = compare_in_window(sp, window_vec)
            if length(window_vec) == sp.window_size
                index_center = (sp.window_size+1)/2;
                value_center = window_vec(index_center);
                value_window = window_vec;
                if sum(abs(value_window - value_center)/value_center < sp.percent_change) >1
                    decision_loop = true;
                else
                    decision_loop = false;
                end
            else
                error("the window size does not match")
            end
        end %end of method loop_decision
        
        function is_already = check_already(sp, iter, criteria) 
            is_already =  sum(sp.stop_history{1:iter, criteria}) > 0;
        end%end of is_already

        function is_early = check_early(sp, iter, standard)
             if iter <= standard
                 is_early = true;
             else
                 is_early = false;
             end
        end%end of method check_early

    end%end of methods
end%end of classdef