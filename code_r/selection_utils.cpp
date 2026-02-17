#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <random>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// [[Rcpp::plugins(openmp)]]

//' High-performance permutation test for greedy screening
//' 
//' @param X Data matrix (p x n)
//' @param obs_stat Observed statistics vector (length p)
//' @param indicator Base indicator vector (length n)
//' @param factor1 Scalar factor1 = (1/n1 + 1/n2)
//' @param factor2 Vector factor2 = SumTotal / n2 (length p)
//' @param n_perms Number of permutations
//' @return Vector of counts (length p) where perm_stat >= obs_stat
// [[Rcpp::export]]
IntegerVector fast_perm_test_cpp(NumericMatrix X, NumericVector obs_stat, 
                                 IntegerVector indicator, double factor1, 
                                 NumericVector factor2, int n_perms) {
    int p = X.nrow();
    int n = X.ncol();
    
    // Result counts initialized to zero
    IntegerVector global_counts(p, 0);
    
    #ifdef _OPENMP
    #pragma omp parallel
    {
        // Thread-local variables
        std::vector<int> local_indicator(n);
        for(int i = 0; i < n; ++i) local_indicator[i] = indicator[i];
        
        std::vector<int> local_counts(p, 0);
        
        // Thread-local random engine
        // Use a seed that depends on thread ID for independence
        int tid = omp_get_thread_num();
        std::mt19937 g(std::random_device{}() + tid);
        
        #pragma omp for
        for (int b = 0; b < n_perms; ++b) {
            // 1. Shuffle indicator
            std::shuffle(local_indicator.begin(), local_indicator.end(), g);
            
            // 2. Iterate over features
            for (int i = 0; i < p; ++i) {
                double sum1 = 0.0;
                // We only need Sum1 (where indicator == 1)
                for (int j = 0; j < n; ++j) {
                    if (local_indicator[j] == 1) {
                        sum1 += X(i, j);
                    }
                }
                
                // 3. Compute stat: |Sum1 * (1/n1 + 1/n2) - Sum_total/n2|
                double perm_stat = std::abs(sum1 * factor1 - factor2[i]);
                
                if (perm_stat >= obs_stat[i]) {
                    local_counts[i] += 1;
                }
            }
        }
        
        // Merge local counts to global counts safely
        #pragma omp critical
        {
            for (int i = 0; i < p; ++i) {
                global_counts[i] += local_counts[i];
            }
        }
    }
    #else
    // Sequential fallback
    std::vector<int> local_indicator(n);
    for(int i = 0; i < n; ++i) local_indicator[i] = indicator[i];
    std::mt19937 g(std::random_device{}());
    
    for (int b = 0; b < n_perms; ++b) {
        std::shuffle(local_indicator.begin(), local_indicator.end(), g);
        for (int i = 0; i < p; ++i) {
            double sum1 = 0.0;
            for (int j = 0; j < n; ++j) {
                if (local_indicator[j] == 1) sum1 += X(i, j);
            }
            double perm_stat = std::abs(sum1 * factor1 - factor2[i]);
            if (perm_stat >= obs_stat[i]) global_counts[i] += 1;
        }
    }
    #endif
    
    return global_counts;
}

// Manual wrapper for R CMD SHLIB / .Call interface
extern "C" SEXP fast_perm_test_wrapper(SEXP XSEXP, SEXP obs_statSEXP, 
                                      SEXP indicatorSEXP, SEXP factor1SEXP, 
                                      SEXP factor2SEXP, SEXP n_permsSEXP) {
    NumericMatrix X(XSEXP);
    NumericVector obs_stat(obs_statSEXP);
    IntegerVector indicator(indicatorSEXP);
    double factor1 = as<double>(factor1SEXP);
    NumericVector factor2(factor2SEXP);
    int n_perms = as<int>(n_permsSEXP);
    
    return Rcpp::wrap(fast_perm_test_cpp(X, obs_stat, indicator, factor1, factor2, n_perms));
}
