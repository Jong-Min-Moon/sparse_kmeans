n_subsample = 45;
file_x = "leuk_x.txt";
file_y = "leuk_y.txt";
x = readmatrix(file_x);
x = 2.^x;
x = normalize(x');
x = x';

gen = data_generator_subsample(selected_data', selected_integer_labels);
[x_new, y_new] = gen.get_data(n_subsample, 1);
n_rep = 100;
acc = repelem(0,n_rep);
n_iter=20;
for i=1:n_iter
    gen = data_generator_subsample(selected_data', selected_integer_labels);
    [x_new, y_new] = gen.get_data(n_subsample, i);
    sol_ifpca = ifpca(x_new, 2);
    acc(i) = get_bicluster_accuracy(sol_ifpca, y_new);
end

mean(acc)