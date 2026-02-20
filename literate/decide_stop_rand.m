function is_stop = decide_stop_rand(rand_vec, loop_detect_start, window_size, min_delta)
 is_stop = false;
        % Early stopping logic
        stop_rand = detect_rand_1(rand_vec, window_size);
        %stagnate_rand = detect_loop(rand_vec, loop_detect_start, window_size, min_delta);
        stagnate_rand = false;
        flags = [stop_rand, stagnate_rand];
        flag_names = {'stop_rand', 'stagnate_rand'};
        if sum(flags) >= 1
            fprintf('\nStopping early. Activated conditions:\n');
            for i = 1:length(flags)
                if flags(i)
                    fprintf('  • %s\n', flag_names{i});
                end
            end
            is_stop = true;
        end
end
