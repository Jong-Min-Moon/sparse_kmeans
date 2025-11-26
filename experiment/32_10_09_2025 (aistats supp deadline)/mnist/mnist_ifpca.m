n_subsample = 1000;
load('mnista_data.mat'); % get data from attatched autoencoder.ipynb
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