classdef stopper < handle
    properties
        window_size
        percent_change
        max_iter
    end
    methods
        function sp = stopper(window_size, percent_change, max_iter)
            sp.window_size = window_size;
            sp.percent_change = percent_change;
            sp.max_iter = max_iter;
        end

        function criteria_vec = pull_criteria(sp, obj_val_original, obj_val_sdp, stopping_criteria_vec, iter)
            criteria_vec = dictionary(["original", "sdp", "loop"], [false,false,false]);
            if iter>1
                criteria_vec("original") = sp.detect_relative_change(obj_val_original, stopping_criteria_vec, iter, "original");
                criteria_vec("sdp")      = sp.detect_relative_change(obj_val_sdp     , stopping_criteria_vec, iter, "sdp");
                criteria_vec("loop")     = sp.detect_loop(obj_val_original, obj_val_sdp, stopping_criteria_vec, iter)
            end
        end

        function is_converge = detect_relative_change(sp, obj_val_vec, stopping_criteria_vec, iter, criteria)
            if (sp.check_already(stopping_criteria_vec, iter, criteria) | (sp.get_relative_change(obj_val_vec, iter) > sp.percent_change))
                is_converge = false;
            else
                is_converge = true;
            end                    
        end %end of detect_relative_change
        function relative_change = get_relative_change(sp, obj_val_vec, iter)
            relative_change = abs((obj_val_vec(iter) - obj_val_vec(iter-1))/obj_val_vec(iter-1));
        end

        function is_loop = detect_loop(sp, obj_val_original, obj_val_sdp, stopping_criteria_vec, iter)
            if (sp.check_early(iter, (sp.window_size-1)/2) | sp.check_already(stopping_criteria_vec, iter, "loop"))
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
        
        function is_already = check_already(sp, stopping_criteria_vec, iter, criteria) 
            is_already =  sum(ismember(stopping_criteria_vec(1:iter), criteria)) > 0;
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