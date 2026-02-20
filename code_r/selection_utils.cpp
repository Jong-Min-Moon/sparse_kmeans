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

// [[Rcpp::export]]
IntegerVector sam_perm_test_cpp(NumericMatrix X, IntegerVector indicator, 
                                double factor1, NumericVector factor2, 
                                int n_perms, NumericVector thresholds) {
    int p = X.nrow();
    int n = X.ncol();
    int n_thresh = thresholds.size();
    
    // Result: For each threshold, how many null stats >= threshold?
    IntegerVector global_counts(n_thresh, 0);
    
    #ifdef _OPENMP
    #pragma omp parallel
    {
        // Thread-local copies
        std::vector<int> local_indicator(n);
        for(int i = 0; i < n; ++i) local_indicator[i] = indicator[i];
        
        std::vector<int> local_counts(n_thresh, 0);
        
        int tid = omp_get_thread_num();
        std::mt19937 g(std::random_device{}() + tid);
        
        // Scratch space for stats (avoid reallocating)
        std::vector<double> perm_stats(p);

        #pragma omp for
        for (int b = 0; b < n_perms; ++b) {
            // 1. Shuffle
            std::shuffle(local_indicator.begin(), local_indicator.end(), g);
            
            // 2. Compute Stats
            for (int i = 0; i < p; ++i) {
                double sum1 = 0.0;
                for (int j = 0; j < n; ++j) {
                    if (local_indicator[j] == 1) sum1 += X(i, j);
                }
                perm_stats[i] = std::abs(sum1 * factor1 - factor2[i]);
            }
            
            // 3. Sort Stats Descending
            std::sort(perm_stats.begin(), perm_stats.end(), std::greater<double>());
            
            // 4. Two-pointer match against Thresholds (also Descending)
            // thresholds[k] vs perm_stats[j]
            // We want count of perm_stats >= thresholds[k]
            
            int stats_idx = 0;
            for (int k = 0; k < n_thresh; ++k) {
                double t = thresholds[k];
                // Advance stats pointer while stats >= t
                while (stats_idx < p && perm_stats[stats_idx] >= t) {
                    stats_idx++;
                }
                // stats_idx is now the count of stats >= t
                local_counts[k] += stats_idx; 
            }
        }
        
        #pragma omp critical
        {
            for (int k = 0; k < n_thresh; ++k) {
                global_counts[k] += local_counts[k];
            }
        }
    }
    #else
    // Serial Fallback
    std::vector<int> local_indicator(n);
    for(int i = 0; i < n; ++i) local_indicator[i] = indicator[i];
    std::mt19937 g(std::random_device{}());
    std::vector<double> perm_stats(p);
    
    for (int b = 0; b < n_perms; ++b) {
        std::shuffle(local_indicator.begin(), local_indicator.end(), g);
        for (int i = 0; i < p; ++i) {
            double sum1 = 0.0;
            for (int j = 0; j < n; ++j) {
                if (local_indicator[j] == 1) sum1 += X(i, j);
            }
            perm_stats[i] = std::abs(sum1 * factor1 - factor2[i]);
        }
        std::sort(perm_stats.begin(), perm_stats.end(), std::greater<double>());
        
        int stats_idx = 0;
        for (int k = 0; k < n_thresh; ++k) {
            double t = thresholds[k];
            while (stats_idx < p && perm_stats[stats_idx] >= t) stats_idx++;
            global_counts[k] += stats_idx;
        }
    }
    #endif
    
    return global_counts;
}

extern "C" SEXP sam_perm_test_wrapper(SEXP XSEXP, SEXP indicatorSEXP, 
                                     SEXP factor1SEXP, SEXP factor2SEXP, 
                                     SEXP n_permsSEXP, SEXP thresholdsSEXP) {
    NumericMatrix X(XSEXP);
    IntegerVector indicator(indicatorSEXP);
    double factor1 = as<double>(factor1SEXP);
    NumericVector factor2(factor2SEXP);
    int n_perms = as<int>(n_permsSEXP);
    NumericVector thresholds(thresholdsSEXP);
    
    return Rcpp::wrap(sam_perm_test_cpp(X, indicator, factor1, factor2, n_perms, thresholds));
}

