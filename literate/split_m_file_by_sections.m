function split_m_file_by_sections(filename)
% split_m_file_by_sections - Extracts sections from a .m file marked by '%%',
% and exports each function or class marked with @export to its own file.
%
% For each exported section, the function removes any block that starts with
% "% *Example:*" followed by lines of code (non-comment, non-empty lines).
%
%   Inputs:
%       filename - Path to the .m file to be processed.
%
%   Outputs:
%       Creates a new directory named '[filename]_sections' in the same
%       location as the input file, containing the split function/class files.

    % Check that file exists
    if ~isfile(filename)
        error('split_m_file_by_sections:FileNotFound', 'File not found: %s', filename);
    end

    % Read lines from the .m file
    fid = fopen(filename, 'r');
    if fid == -1
        error('split_m_file_by_sections:FileOpenError', 'Could not open file: %s', filename);
    end
    lines = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
    fclose(fid);
    lines = lines{1};

    sections = {};
    current_section = {};
    inside_section = false;

    % --- Section Parsing ---
    % Sections are defined by '%% ' (two percent signs and a space)
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

    % Add the final section if it exists
    if inside_section && ~isempty(current_section)
        sections{end+1} = current_section;
    end

    % Output folder
    [folder, base, ~] = fileparts(filename);
    output_dir = fullfile(folder, [base '_sections']);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % --- Process Each Section ---
    for i = 1:length(sections)
        section_lines = sections{i};

        % Section must contain @export to be processed
        if ~any(contains(section_lines, '@export'))
            continue;
        end

        % Find the first non-comment, non-empty line that could be a function or class definition
        definition_idx = find(~startsWith(strtrim(section_lines), '%') & ~cellfun('isempty', strtrim(section_lines)), 1, 'first');

        if isempty(definition_idx)
            warning('split_m_file_by_sections:NoDefinitionFound', 'Skipping section %d: No function or class definition found.', i);
            continue;
        end

        first_code_line = strtrim(section_lines{definition_idx});

        func_or_class_name = '';
        is_class_def = false;
        
        % Check for classdef
        if startsWith(first_code_line, 'classdef')
            is_class_def = true;
            % Extract class name: classdef ClassName < SuperClass
            tokens = regexp(first_code_line, 'classdef\s+(\w+)', 'tokens', 'once');
            if ~isempty(tokens)
                func_or_class_name = tokens{1};
            end
        % Check for function
        elseif startsWith(first_code_line, 'function')
            % Extract function name (handling output arguments)
            tokens = regexp(first_code_line, 'function\s+(?:\[[^\]]*\]|\S+)?\s*=\s*(\w+)', 'tokens', 'once');
            if isempty(tokens)
                % Handle functions without output arguments
                tokens = regexp(first_code_line, 'function\s+(\w+)\s*\(', 'tokens', 'once');
            end
            if ~isempty(tokens)
                func_or_class_name = tokens{1};
            end
        end

        if isempty(func_or_class_name)
            warning('split_m_file_by_sections:NameParseError', 'Could not parse function/class name from line: %s in section %d. Skipping.', first_code_line, i);
            continue;
        end

        % Prepare lines for output file
        final_output_lines = section_lines; % Start with the full section

        % If it's a class definition, ensure 'classdef' is at the very top
        % For functions, the 'function' line is already handled correctly by
        % moving it to the top if it wasn't the first code line.
        if is_class_def && definition_idx ~= 1
            % Move the classdef line to the very beginning of the section_lines
            classdef_line = final_output_lines{definition_idx};
            final_output_lines(definition_idx) = []; % Remove from its original spot
            final_output_lines = [{classdef_line}; final_output_lines(:)]; % Add to top
        end

        % === Remove "*Example:*" block and following code lines ===
        % This logic applies to both functions and classes
        example_idx = find(contains(final_output_lines, '% *Example:*'), 1, 'first');
        if ~isempty(example_idx)
            cutoff_idx = example_idx;
            % Iterate to find the end of the example block
            for k = example_idx+1:length(final_output_lines)
                line_k_trimmed = strtrim(final_output_lines{k});
                % Stop if it's a comment or empty line, indicating end of code in example
                % IMPORTANT: For classes, this might break the classdef block
                % if an example is mid-class. The assumption is examples are at the end.
                if startsWith(line_k_trimmed, '%') || isempty(line_k_trimmed)
                    break;
                end
                cutoff_idx = k;
            end
            % Remove the example block (from '% *Example:*' to the last code line)
            final_output_lines(example_idx:cutoff_idx) = [];
        end

        % Write to .m file
        out_filename = fullfile(output_dir, [func_or_class_name '.m']);
        fid_out = fopen(out_filename, 'w');
        if fid_out == -1
            warning('split_m_file_by_sections:FileWriteError', 'Could not write to file: %s. Skipping.', out_filename);
            continue;
        end
        for j = 1:length(final_output_lines)
            fprintf(fid_out, '%s\n', final_output_lines{j});
        end
        fclose(fid_out);
        fprintf('Saved: %s\n', out_filename);
    end
    fprintf('Finished splitting sections for %s. Files saved to %s\n', filename, output_dir);
end