function is_stop = decide_stop(obj_sdp, obj_lik, loop_detect_start, window_size, min_delta)
%% decide_stop
% @export
 is_stop = false;
        % Early stopping logic
        stop_sdp = detect_relative_change(obj_sdp, loop_detect_start, min_delta);
        stop_lik = detect_relative_change(obj_lik, loop_detect_start, min_delta);
        stagnate_sdp = detect_loop(obj_sdp, loop_detect_start, window_size, min_delta);
        stagnate_lik = detect_loop(obj_lik, loop_detect_start, window_size, min_delta);
        flags = [stop_sdp, stop_lik, stagnate_sdp, stagnate_lik];
        flag_names = {'stop_sdp', 'stop_lik', 'stagnate_sdp', 'stagnate_lik'};
        if sum(flags) >= 2
            fprintf('\nStopping early. Activated conditions:\n');
            for i = 1:length(flags)
                if flags(i)
                    fprintf('  • %s\n', flag_names{i});
                end
            end
            is_stop = true;
        end
 end
%% 
%% 
