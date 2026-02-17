
#include <Rcpp.h>
#include <algorithm>
#include <vector>

using namespace Rcpp;

// [[Rcpp::interfaces(r, cpp)]]



//' Project each row of a matrix onto the probability simplex
//'
//' Solves: min ||x - y||^2 s.t. y >= 0, sum(y) = 1
//' for each row x of Mat.
//'
//' @param Mat Input matrix
//' @return Matrix with projected rows
// [[Rcpp::export]]
NumericMatrix proj_simplex_rows_cpp(NumericMatrix Mat) {
  int n_rows = Mat.nrow();
  int n_cols = Mat.ncol();
  NumericMatrix Y(n_rows, n_cols);

  // Buffer for row data to avoid repeated allocation
  std::vector<double> u(n_cols);

  for (int i = 0; i < n_rows; ++i) {
    // Copy row i to buffer
    for (int j = 0; j < n_cols; ++j) {
      u[j] = Mat(i, j);
    }

    // Sort u in descending order
    // Note: sorting a copy to preserve original indices for reconstruction?
    // Wait, the algorithm needs sorted values to find rho, but the projection
    // y_j = max(x_j + theta, 0) depends on original x_j.
    // So we need a sorted copy 'sorted_u'.
    std::vector<double> sorted_u = u;
    std::sort(sorted_u.begin(), sorted_u.end(), std::greater<double>());

    // Find rho
    // cumsum
    double cssv = 0.0;
    int rho_idx = -1; // Indices are 0-based in C++, but logic uses count k=1..p

    for (int j = 0; j < n_cols; ++j) {
      cssv += sorted_u[j];
      // Condition: u_j + (1 - cssv_j) / (j+1) > 0
      double cond = sorted_u[j] + (1.0 - cssv) / (j + 1);
      if (cond > 0.0) {
        rho_idx = j;
      }
    }

    // rho is the count = rho_idx + 1
    // cssv at rho_idx
    double cssv_rho = 0.0;
    for (int j = 0; j <= rho_idx; ++j) {
      cssv_rho += sorted_u[j];
    }
    
    double theta = (1.0 - cssv_rho) / (rho_idx + 1);

    // Construct solution
    for (int j = 0; j < n_cols; ++j) {
      double val = u[j] + theta;
      Y(i, j) = (val > 0.0) ? val : 0.0;
    }
  }

  return Y;
}

// Manual wrapper for R CMD SHLIB / .Call interface
extern "C" SEXP proj_simplex_rows_wrapper(SEXP MatSEXP) {
    NumericMatrix Mat(MatSEXP);
    return Rcpp::wrap(proj_simplex_rows_cpp(Mat));
}
