function loglik = loglik_glasso(Omega, sample)
  n = size(sample,1);
  sample_cov = sample' * sample/n;
  logdet = log( det(Omega) );
  ip = sum(Omega .* sample_cov, "all");
  loglik  = logdet - ip;

