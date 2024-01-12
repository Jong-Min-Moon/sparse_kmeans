function cluster_init = ESSC(x, K)
n = size(x,1);
p = size(x,2);
H_hat = (x * x');
[V,D] = eig(H_hat);
[d,ind] = sort(diag(D));
Ds = D(ind,ind);
Vs = V(:,ind);

t_1_hat = Ds(1,1);
t_2_hat = Ds(2,2);
u_1_hat = Vs(:,1);
u_2_hat = Vs(:,2);

tau_n = 1/log(n+p)
delta_n = tau_n^2
f_1 = n^(-0.5)*abs(sum(u_1_hat))-1;


if t_1_hat/t_2_hat < 1+ tau_n
    selected_eigen_idx= [1,2];
elseif abs(f_1) >= delta_n
    selected_eigen_idx = [1];
else
    selected_eigen_idx = [2];
end
U_hat = Vs(:,selected_eigen_idx);
[idx,C] = kmeans(U_hat,K);
cluster_init = idx .* (idx ~= 2) + (idx == 2)* (-1);
cluster_init = cluster_init'
