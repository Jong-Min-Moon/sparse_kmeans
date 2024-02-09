function bic_val= bic(covariance, precision, n_samples, n_features)


    l_theta = -sum(covariance * precision,"all") + logdet(precision);
    l_theta = l_theta * n_features / 2;
    mask = abs(precision)>0.01;
    precision_nnz = (sum(mask, "all") - n_features) / 2.0 ; % lower off diagonal tri

    bic_val=-2.0 * l_theta + precision_nnz * log(n_samples);
   
