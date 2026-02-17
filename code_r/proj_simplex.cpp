
#include <Rcpp.h>
#include <algorithm>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// [[Rcpp::interfaces(r, cpp)]]



//' Project each row of a matrix onto the simplex with variable target sum
//'
//' Algorithm: Randomized selection (expected O(N)) adapted from Duchi et al. (2008).
//' Uses value-based partitioning to guarantee convergence.
//'
//' @param Mat Input matrix
//' @param target_sum Target sum for each row (default 1.0)
//' @return Matrix with projected rows
// [[Rcpp::export]]
//' @param Mat Input matrix
//' @param target_sum Target sum for each row (default 1.0)
//' @return Matrix with projected rows
// [[Rcpp::export]]
NumericMatrix proj_simplex_rows_cpp(NumericMatrix Mat, double target_sum = 1.0) {
  int n_rows = Mat.nrow();
  int n_cols = Mat.ncol();
  NumericMatrix Y(n_rows, n_cols);

  // Parallel Region: Allocate workspace once per thread
  #ifdef _OPENMP
  #pragma omp parallel
  {
  #endif
    // Thread-local buffer for row data
    std::vector<double> u(n_cols);

    #ifdef _OPENMP
    #pragma omp for
    #endif
    for (int i = 0; i < n_rows; ++i) {
      // Copy row i to buffer
      for (int j = 0; j < n_cols; ++j) {
        u[j] = Mat(i, j);
      }

      // Sort u in descending order
      std::sort(u.begin(), u.end(), std::greater<double>());

      // Find rho
      double cssv = 0.0;
      int rho_idx = -1; 

      for (int j = 0; j < n_cols; ++j) {
        cssv += u[j];
        // Condition: u_j + (target_sum - cssv_j) / (j+1) > 0
        double cond = u[j] + (target_sum - cssv) / (j + 1);
        if (cond > 0.0) {
          rho_idx = j;
        }
      }

      // cssv at rho_idx
      double cssv_rho = 0.0;
      // Optimize: redundant summation, can we reuse cssv?
      // No, cssv in loop went up to n_cols. 
      // But we can just sum again or store prefix sums.
      // Re-summing 0..rho_idx is fast.
      for (int j = 0; j <= rho_idx; ++j) {
        cssv_rho += u[j];
      }
      
      double theta = (target_sum - cssv_rho) / (rho_idx + 1);

      // Construct solution
      for (int j = 0; j < n_cols; ++j) {
        double val = Mat(i, j) + theta;
        Y(i, j) = (val > 0.0) ? val : 0.0;
      }
    }
  #ifdef _OPENMP
  }
  #endif

  return Y;
}

// Manual wrapper for R CMD SHLIB / .Call interface
extern "C" SEXP proj_simplex_rows_wrapper(SEXP MatSEXP, SEXP SumSEXP) {
    NumericMatrix Mat(MatSEXP);
    double target_sum = as<double>(SumSEXP);
    return Rcpp::wrap(proj_simplex_rows_cpp(Mat, target_sum));
}
