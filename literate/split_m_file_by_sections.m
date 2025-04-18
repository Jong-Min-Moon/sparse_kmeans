function split_m_file_by_sections(filename)
% split_m_file_by_sections - Extracts sections from a .m file marked by '%%',
% and exports each function marked with @export to its own file.
%
% For each exported section, the function removes any block that starts with
% "% *Example:*" followed by lines of code (non-comment, non-empty lines).

    % Check that file exists
    if ~isfile(filename)
        error('File not found: %s', filename);
    end

    % Read lines from the .m file
    fid = fopen(filename, 'r');
    lines = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
    fclose(fid);
    lines = lines{1};

    sections = {};
    current_section = {};
    inside_section = false;

    for i = 1:length(lines)
        line = lines{i};
        stripped = strtrim(line);

        % Start of a new titled section
        if startsWith(stripped, '%% ')
            if inside_section && ~isempty(current_section)
                sections{end+1} = current_section; %#ok<AGROW>
            end
            current_section = {line};
            inside_section = true;
        else
            if inside_section
                current_section{end+1} = line; %#ok<AGROW>
            end
        end
    end

    % Final section
    if inside_section && ~isempty(current_section)
        sections{end+1} = current_section;
    end

    % Output folder
    [folder, base, ~] = fileparts(filename);
    output_dir = fullfile(folder, [base '_sections']);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Process sections
    for i = 1:length(sections)
        section_lines = sections{i};

        % Must contain @export
        if ~any(contains(section_lines, '@export'))
            continue;
        end

        % Skip comments when looking for the function line
        func_idx = find(~startsWith(strtrim(section_lines), '%') & contains(strtrim(section_lines), 'function'), 1, 'first');
        if isempty(func_idx)
            warning('Skipping section without function definition.');
            continue;
        end

        func_line = strtrim(section_lines{func_idx});
        % Extract function name using regex
        tokens = regexp(func_line, 'function\s+(?:\[[^\]]*\]|\S+)?\s*=\s*(\w+)', 'tokens');
        if isempty(tokens)
            tokens = regexp(func_line, 'function\s+(\w+)\s*\(', 'tokens');
        end
        if isempty(tokens)
            warning('Could not parse function name from line: %s', func_line);
            continue;
        end
        func_name = tokens{1}{1};

        % Reorder function line to the top
        remaining_lines = section_lines;
        remaining_lines(func_idx) = [];
        section_lines = [{func_line}; remaining_lines(:)];

        % === Remove "*Example:*" block and following code lines ===
        example_idx = find(contains(section_lines, '% *Example:*'), 1, 'first');
        if ~isempty(example_idx)
            cutoff_idx = example_idx;
            for k = example_idx+1:length(section_lines)
                line_k = strtrim(section_lines{k});
                if startsWith(line_k, '%') || isempty(line_k)
                    break;
                end
                cutoff_idx = k;
            end
            section_lines(example_idx:cutoff_idx) = [];
        end

        % Write to .m file
        out_filename = fullfile(output_dir, [func_name '.m']);
        fid_out = fopen(out_filename, 'w');
        for j = 1:length(section_lines)
            fprintf(fid_out, '%s\n', section_lines{j});
        end
        fclose(fid_out);
        fprintf('Saved: %s\n', out_filename);
    end
end
