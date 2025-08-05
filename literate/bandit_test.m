p=5000;
n=200;
s=10
sigma = 1;
K=2;
sep=4;
rep=111;
cluster_1_ratio=0.5;
%[data, label_true, mu1, mu2, generated_sep, ~, beta_star] = generate_gaussian_data(n, p, 10, sep, 'iso', 'equal_symmetric', 0, rep, cluster_1_ratio, rep);

%%

 
%%
aaa = sdp_kmeans_bandit_thinning_nmf_simul(data', 2);
aaa.fit_predict(100, label_true')
%%
 
 gen = data_generator_approximately_sparse_mean(n, p, s, sep, rep, cluster_1_ratio)
 [data_2, label_true_2]       = gen.get_data(1, 0);
bbb = sdp_kmeans_bandit_thinning_nmf_simul(data_2, 2);
bbb.fit_predict(300, label_true_2)