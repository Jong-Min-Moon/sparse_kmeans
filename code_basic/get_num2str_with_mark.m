function num2str_with_mark = get_num2str_with_mark(integer_vec, mark)
    num2str_with_mark = regexprep(num2str(integer_vec),'\s+',mark);
end