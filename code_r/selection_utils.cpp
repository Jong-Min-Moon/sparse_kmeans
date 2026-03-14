#include <Rcpp.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

#include <cmath>
#include <limits>

inline double safe_log_abs(double x) {
    double ax = std::abs(x);
    if (ax == 0.0)
        return -std::numeric_limits<double>::infinity();
    return std::log(ax);
}

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
List fast_perm_test_cpp(NumericMatrix X, NumericVector obs_stat,
                                 IntegerVector indicator, double factor1,
                                 NumericVector factor2, int n_perms, double p_val_threshold) {
  int p = X.nrow();
  int n = X.ncol();

  // Result counts initialized to zero
  std::vector<int> global_counts(p, 0);
  std::vector<std::vector<double>> all_perm_stats(p, std::vector<double>(n_perms));

#ifdef _OPENMP
#pragma omp parallel
  {
    // Thread-local variables
    std::vector<int> local_indicator(n);
    for (int i = 0; i < n; ++i)
      local_indicator[i] = indicator[i];

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
        all_perm_stats[i][b] = perm_stat;

        double log_perm = safe_log_abs(perm_stat);
        double log_obs = safe_log_abs(obs_stat[i]);

        if (log_perm >= log_obs) {
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
  for (int i = 0; i < n; ++i)
    local_indicator[i] = indicator[i];
  std::mt19937 g(std::random_device{}());

  for (int b = 0; b < n_perms; ++b) {
    std::shuffle(local_indicator.begin(), local_indicator.end(), g);
    for (int i = 0; i < p; ++i) {
      double sum1 = 0.0;
      for (int j = 0; j < n; ++j) {
        if (local_indicator[j] == 1)
          sum1 += X(i, j);
      }
      double perm_stat = std::abs(sum1 * factor1 - factor2[i]);
      all_perm_stats[i][b] = perm_stat;
      double log_perm = safe_log_abs(perm_stat);
      double log_obs = safe_log_abs(obs_stat[i]);
      if (log_perm >= log_obs)
        global_counts[i] += 1;
    }
  }
#endif

  NumericVector p_values(p);
  NumericVector percentile_values(p);

  double q = 1.0 - p_val_threshold;
  int k = static_cast<int>(std::ceil(q * n_perms)) - 1;
  k = std::max(0, std::min(k, n_perms - 1));

  for (int i = 0; i < p; ++i) {
    p_values[i] = (global_counts[i] + 1.0) / (n_perms + 1.0);
    
    std::nth_element(
        all_perm_stats[i].begin(),
        all_perm_stats[i].begin() + k,
        all_perm_stats[i].end()
    );
    percentile_values[i] = all_perm_stats[i][k];
  }

  return List::create(Named("p_value") = p_values,
                      Named("percentile_value") = percentile_values);
}

// [[Rcpp::export]]
NumericMatrix get_signed_perm_stats_cpp(NumericMatrix X,
                                        IntegerVector indicator, double factor1,
                                        NumericVector factor2, int n_perms) {
  int p = X.nrow();
  int n = X.ncol();

  // Result: (p x n_perms) matrix of signed raw statistics
  NumericMatrix perm_stats(p, n_perms);

#ifdef _OPENMP
#pragma omp parallel
  {
    // Thread-local copies
    std::vector<int> local_indicator(n);
    for (int i = 0; i < n; ++i)
      local_indicator[i] = indicator[i];

    int tid = omp_get_thread_num();
    std::mt19937 g(std::random_device{}() + tid);

#pragma omp for
    for (int b = 0; b < n_perms; ++b) {
      // 1. Shuffle
      std::shuffle(local_indicator.begin(), local_indicator.end(), g);

      // 2. Compute Stats
      for (int i = 0; i < p; ++i) {
        double sum1 = 0.0;
        for (int j = 0; j < n; ++j) {
          if (local_indicator[j] == 1)
            sum1 += X(i, j);
        }
        // Signed raw difference: mean1 - mean2
        // = sum1 * factor1 - factor2[i]
        perm_stats(i, b) = sum1 * factor1 - factor2[i];
      }
    }
  }
#else
  std::vector<int> local_indicator(n);
  for (int i = 0; i < n; ++i)
    local_indicator[i] = indicator[i];
  std::mt19937 g(std::random_device{}());

  for (int b = 0; b < n_perms; ++b) {
    std::shuffle(local_indicator.begin(), local_indicator.end(), g);
    for (int i = 0; i < p; ++i) {
      double sum1 = 0.0;
      for (int j = 0; j < n; ++j) {
        if (local_indicator[j] == 1)
          sum1 += X(i, j);
      }
      perm_stats(i, b) = sum1 * factor1 - factor2[i];
    }
  }
#endif

  return perm_stats;
}

// [[Rcpp::export]]
IntegerVector count_matrix_exceedances_cpp(NumericMatrix stats_matrix,
                                           NumericVector thresholds) {
  int p = stats_matrix.nrow();
  int n_perms = stats_matrix.ncol();
  int n_thresh = thresholds.size();

  IntegerVector global_counts(n_thresh, 0);

  // Convert matrix to vector for easier iteration if needed, but column-access
  // is fine. We iterate per threshold, then per permutation? Or per
  // permutation, then per threshold. Iterating per permutation allows sorting
  // the column for fast threshold lookup (log p). Or just linear scan if p is
  // small. For large p, sorting each column is O(p log p). 2-pointer is O(p +
  // T). Since we do this for every permutation, efficient counting is key.

#ifdef _OPENMP
#pragma omp parallel
  {
    std::vector<int> local_counts(n_thresh, 0);
    std::vector<double> col_stats(p);

#pragma omp for
    for (int b = 0; b < n_perms; ++b) {
      // Copy column
      for (int i = 0; i < p; ++i)
        col_stats[i] = stats_matrix(i, b);

      // Sort descending
      std::sort(col_stats.begin(), col_stats.end(), std::greater<double>());

      // Match against sorted thresholds (descending)
      int stats_idx = 0;
      for (int k = 0; k < n_thresh; ++k) {
        double t = thresholds[k];
        while (stats_idx < p && col_stats[stats_idx] >= t) {
          stats_idx++;
        }
        local_counts[k] += stats_idx;
      }
    }

#pragma omp critical
    {
      for (int k = 0; k < n_thresh; ++k)
        global_counts[k] += local_counts[k];
    }
  }
#else
  std::vector<double> col_stats(p);
  for (int b = 0; b < n_perms; ++b) {
    for (int i = 0; i < p; ++i)
      col_stats[i] = stats_matrix(i, b);
    std::sort(col_stats.begin(), col_stats.end(), std::greater<double>());
    int stats_idx = 0;
    for (int k = 0; k < n_thresh; ++k) {
      double t = thresholds[k];
      while (stats_idx < p && col_stats[stats_idx] >= t)
        stats_idx++;
      global_counts[k] += stats_idx;
    }
  }
#endif

  return global_counts;
}

extern "C" SEXP get_signed_perm_stats_wrapper(SEXP XSEXP, SEXP indicatorSEXP,
                                              SEXP factor1SEXP,
                                              SEXP factor2SEXP,
                                              SEXP n_permsSEXP) {
  NumericMatrix X(XSEXP);
  IntegerVector indicator(indicatorSEXP);
  double factor1 = as<double>(factor1SEXP);
  NumericVector factor2(factor2SEXP);
  int n_perms = as<int>(n_permsSEXP);

  return Rcpp::wrap(
      get_signed_perm_stats_cpp(X, indicator, factor1, factor2, n_perms));
}

extern "C" SEXP count_matrix_exceedances_wrapper(SEXP statsSEXP,
                                                 SEXP threshSEXP) {
  NumericMatrix stats(statsSEXP);
  NumericVector thresh(threshSEXP);
  return Rcpp::wrap(count_matrix_exceedances_cpp(stats, thresh));
}

extern "C" SEXP fast_perm_test_wrapper(SEXP XSEXP, SEXP obs_statSEXP,
                                       SEXP indicatorSEXP, SEXP factor1SEXP,
                                       SEXP factor2SEXP, SEXP n_permsSEXP,
                                       SEXP p_val_thresholdSEXP) {
  NumericMatrix X(XSEXP);
  NumericVector obs_stat(obs_statSEXP);
  IntegerVector indicator(indicatorSEXP);
  double factor1 = as<double>(factor1SEXP);
  NumericVector factor2(factor2SEXP);
  int n_perms = as<int>(n_permsSEXP);
  double p_val_threshold = as<double>(p_val_thresholdSEXP);

  return Rcpp::wrap(
      fast_perm_test_cpp(X, obs_stat, indicator, factor1, factor2, n_perms, p_val_threshold));
}
