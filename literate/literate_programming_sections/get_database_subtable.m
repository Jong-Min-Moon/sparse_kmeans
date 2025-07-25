%% get_database_subtable
% @export
function database_subtable = get_database_subtable(rep, Delta, support, obj, acc)
            s = length(support);
            current_time = get_current_time();
            
 
             
            n_row = 0;
            dummy = repelem(0, n_row+1)';
            database_subtable = table(...
                repelem(rep, n_row+1)',...                      % 01 replication number
                (1:(n_row+1))',...                              % 02 step iteration number
                repelem(Delta, n_row+1)',...                    % 03 separation
                repelem(obj.p, n_row+1)',...                    % 04 data dimension
                repelem(obj.n, n_row+1)',...                      % 05 sample size
                repelem(s, n_row+1)',...                        % 06 model
                ...
                repelem(acc, n_row+1)',...             % 07 accuracy
                ...
                dummy,...               % 8 sdp objective function value  
                dummy,...               % 9 likelihood value
                ...
                dummy,...                           % 10 true positive
                dummy,...                          % 11 false positive
                dummy,...                          % 12 false negative
                ...
                repelem(current_time, n_row+1)', ...            % 13 timestamp
                'VariableNames', ...
                ...  %1      2       3      4      5        6         
                ["rep", "iter", "sep", "dim", "n", "model", ...
                ...  %7        8           9                       
                 "acc", "obj_sdp", "obj_lik",  ...
                ... % 10          11            12
                 "true_pos", "false_pos",  "false_neg",...
                ...  13
                     "jobdate"]);
        end % end of get_database_subtable
