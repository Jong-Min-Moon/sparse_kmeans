function support_diff = count_support_diff(m, m_hat)
    n_features = size(m,2);

    m_no_diag = m - diag(diag(m));
    m_hat_no_diag = m_hat- diag(diag(m_hat));

    m_nnz = sum((m_no_diag ~=0), "all");
    m_hat_nnz = sum((m_hat_no_diag ~=0), "all");

    support_equal = (m_no_diag~=0) .* (m_hat_no_diag~=0);
    nnz_intersect = sum(support_equal, "all");
support_diff  = m_nnz + m_hat_nnz - (2 * nnz_intersect);