sep=5;
n=200;
s=10;
acc_vec = [0,0,0,0];
p_vec = [2000, 3000, 4000, 5000];
for i = 1:4
    p = p_vec(i);
    fprintf("dimension: %d" ,p)
    acc=0;
    for rep = 1:200
        generator = data_generator_approximately_sparse_mean(n, p, s, sep, rep, 0.5);
        [x_noisy, cluster_true] = generator.get_data(1, 0);
        cluster_est = randomProjectionKMeans(x_noisy,2, 10);
        acc_now = get_bicluster_accuracy(cluster_est, cluster_true);
        acc = acc + acc_now/200;
    end
    acc_vec(i) = acc;
    fprintf("acc %f", acc(i))
end