#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace Rcpp;

// Estimate Excess Moments for 1D Gaussian Mixture Model
List estimate_excess_moments_cpp(const NumericVector &x) {
  int n = x.length();

  double mu = mean(x);
  NumericVector x_c = x - mu;

  double M2 = 0, M3 = 0, M4 = 0, M5 = 0, M6 = 0;
  for (int i = 0; i < n; i++) {
    double v = x_c[i];
    double v2 = v * v;
    double v3 = v2 * v;
    double v4 = v3 * v;
    double v5 = v4 * v;
    double v6 = v5 * v;

    M2 += v2;
    M3 += v3;
    M4 += v4;
    M5 += v5;
    M6 += v6;
  }
  M2 /= n;
  M3 /= n;
  M4 /= n;
  M5 /= n;
  M6 /= n;

  double sigma2 = M2;
  double X3 = M3;
  double X4 = M4 - 3 * M2 * M2;
  double X5 = M5 - 10 * M3 * M2;
  double X6 = M6 - 15 * M4 * M2 + 30 * M2 * M2 * M2;

  return List::create(Named("mu") = mu, Named("sigma2") = sigma2,
                      Named("X3") = X3, Named("X4") = X4, Named("X5") = X5,
                      Named("X6") = X6);
}

// Polynomial addition
std::vector<double> poly_add(const std::vector<double> &p1,
                             const std::vector<double> &p2) {
  int n = std::max(p1.size(), p2.size());
  std::vector<double> res(n, 0.0);
  for (size_t i = 0; i < p1.size(); i++)
    res[i] += p1[i];
  for (size_t i = 0; i < p2.size(); i++)
    res[i] += p2[i];
  return res;
}

// Polynomial multiplication
std::vector<double> poly_mul(const std::vector<double> &p1,
                             const std::vector<double> &p2) {
  if (p1.empty() || p2.empty())
    return std::vector<double>();
  std::vector<double> res(p1.size() + p2.size() - 1, 0.0);
  for (size_t i = 0; i < p1.size(); i++) {
    for (size_t j = 0; j < p2.size(); j++) {
      res[i + j] += p1[i] * p2[j];
    }
  }
  return res;
}

// Recover Alpha from Moments
std::vector<double> RecoverAlphasFromMoments_cpp(double X3, double X4,
                                                 double X5, double X6,
                                                 double epsilon) {
  // Get R's polyroot
  Function polyroot("polyroot");

  // -X3^2 + X4*y + 2*y^3 = 0 -> coeffs: -X3^2, X4, 0, 2 (lowest to highest
  // degree)
  NumericVector coeffs_y = NumericVector::create(-X3 * X3, X4, 0.0, 2.0);
  ComplexVector roots_y = polyroot(coeffs_y);

  double ymax = -1.0;
  bool found_real = false;
  for (int i = 0; i < roots_y.length(); i++) {
    if (std::abs(roots_y[i].i) <
        1e-8) { // Check if imaginary part is close to zero
      double r = roots_y[i].r;
      if (!found_real || r > ymax) {
        ymax = r;
        found_real = true;
      }
    }
  }

  if (!found_real)
    throw std::runtime_error("No real roots found for ymax");
  if (ymax <= 0)
    ymax = 1e-6;

  double kappa = 1.0 + std::sqrt(std::abs(X4)) / ymax;
  double upper_limit = (1.0 + epsilon / kappa) * ymax;

  // p5(y) construction (coefficients in lowest-to-highest degree order for
  // convolution logic)

  // Term 1: 6 * (2*X3^3 - 3*X3*X4*y + X5*y^2 + 2*X3*y^3)^2
  std::vector<double> t1_base = {2.0 * std::pow(X3, 3), -3.0 * X3 * X4, X5,
                                 2.0 * X3};
  std::vector<double> term1 = poly_mul(t1_base, t1_base);
  for (size_t i = 0; i < term1.size(); i++)
    term1[i] *= 6.0;

  // Term 2: (-4*X3^2 + 3*X4*y + 0*y^2 + 2*y^3)^2 * (-X3^2 + X4*y + 0*y^2 +
  // 2*y^3)
  std::vector<double> t2_base1 = {-4.0 * X3 * X3, 3.0 * X4, 0.0, 2.0};
  std::vector<double> t2_sq = poly_mul(t2_base1, t2_base1);
  std::vector<double> t2_base2 = {-X3 * X3, X4, 0.0, 2.0};
  std::vector<double> term2 = poly_mul(t2_sq, t2_base2);

  std::vector<double> p5_coeffs_std = poly_add(term1, term2);

  // Convert std::vector to NumericVector for polyroot
  NumericVector p5_coeffs(p5_coeffs_std.size());
  for (size_t i = 0; i < p5_coeffs_std.size(); i++)
    p5_coeffs[i] = p5_coeffs_std[i];

  ComplexVector all_roots = polyroot(p5_coeffs);

  std::vector<double> candidate_alphas;
  candidate_alphas.push_back(upper_limit);

  for (int i = 0; i < all_roots.length(); i++) {
    if (std::abs(all_roots[i].i) <
        1e-8) { // Check if imaginary part is close to zero
      double r = all_roots[i].r;
      if (r > 0 && r <= upper_limit + 1e-6) {
        candidate_alphas.push_back(r);
      }
    }
  }

  std::sort(candidate_alphas.begin(), candidate_alphas.end(),
            std::greater<double>());
  return candidate_alphas;
}

List SameMeanRecoverFromMoments_cpp(double mu, double sigma2, double X4,
                                    double X6) {
  if (std::abs(X4) < 1e-12) {
    if (X4 >= 0)
      X4 = 1e-12;
    else
      X4 = -1e-12;
  }

  double delta_sigma2 =
      std::sqrt((4.0 / 3.0) * X4 + (X6 * X6) / (25.0 * X4 * X4));

  double p1 = 0.5 * (1.0 + X6 / (5.0 * X4 * delta_sigma2));
  if (p1 < 0)
    p1 = 0;
  else if (p1 > 1)
    p1 = 1;
  double p2 = 1.0 - p1;

  double sigma1_sq = sigma2 - p2 * delta_sigma2;
  double sigma2_sq = sigma2 + p1 * delta_sigma2;

  if (sigma1_sq < 1e-10)
    sigma1_sq = 1e-10;
  if (sigma2_sq < 1e-10)
    sigma2_sq = 1e-10;

  return List::create(
      Named("comp1") = List::create(Named("p") = p1, Named("mu") = mu,
                                    Named("sigma") = std::sqrt(sigma1_sq)),
      Named("comp2") = List::create(Named("p") = p2, Named("mu") = mu,
                                    Named("sigma") = std::sqrt(sigma2_sq)));
}

List RecoverFromMoments_cpp(double mu, double sigma2, double X3, double X4,
                            double X5, double X6, double epsilon) {
  std::vector<double> alphas =
      RecoverAlphasFromMoments_cpp(X3, X4, X5, X6, epsilon);

  bool found_best = false;
  List best_candidate;

  for (size_t i = 0; i < alphas.size(); i++) {
    double alpha = alphas[i];
    double gamma_num = alpha * alpha * X5 + 2 * std::pow(X3, 3) +
                       2 * std::pow(alpha, 3) * X3 - 3 * X3 * X4 * alpha;
    double gamma_den = 4 * X3 * X3 - 2 * std::pow(alpha, 3) - 3 * X4 * alpha;

    if (std::abs(gamma_den) < 1e-12) {
      if (gamma_den >= 0)
        gamma_den = 1e-12;
      else
        gamma_den = -1e-12;
    }

    double gamma = (1.0 / alpha) * (gamma_num / gamma_den);
    double beta = (1.0 / alpha) * (X3 - 3.0 * alpha * gamma);

    double disc_val = beta * beta + 4.0 * alpha;
    if (disc_val < 0)
      continue;

    double disc = std::sqrt(disc_val);
    double mu1 = (beta - disc) / 2.0;
    double mu2 = (beta + disc) / 2.0;

    if (std::abs(mu2 - mu1) < 1e-12)
      continue;

    double p1 = mu2 / (mu2 - mu1);
    if (p1 < -1e-4 || p1 > 1.0001)
      continue;

    if (p1 < 0)
      p1 = 0;
    else if (p1 > 1)
      p1 = 1;
    double p2 = 1.0 - p1;

    double sigma1_sq = sigma2 - (p1 * mu1 * mu1 + p2 * mu2 * mu2 - mu1 * gamma);
    double sigma2_sq = sigma1_sq + (mu2 - mu1) * gamma;

    if (sigma1_sq < -1e-2 || sigma2_sq < -1e-2)
      continue;

    if (sigma1_sq < 1e-10)
      sigma1_sq = 1e-10;
    if (sigma2_sq < 1e-10)
      sigma2_sq = 1e-10;

    best_candidate = List::create(
        Named("comp1") = List::create(Named("p") = p1, Named("mu") = mu1 + mu,
                                      Named("sigma") = std::sqrt(sigma1_sq)),
        Named("comp2") = List::create(Named("p") = p2, Named("mu") = mu2 + mu,
                                      Named("sigma") = std::sqrt(sigma2_sq)));
    found_best = true;
    break;
  }

  if (!found_best) {
    double alpha = alphas[0];
    double gamma_num = alpha * alpha * X5 + 2 * std::pow(X3, 3) +
                       2 * std::pow(alpha, 3) * X3 - 3 * X3 * X4 * alpha;
    double gamma_den = 4 * X3 * X3 - 2 * std::pow(alpha, 3) - 3 * X4 * alpha;

    if (std::abs(gamma_den) < 1e-12) {
      if (gamma_den >= 0)
        gamma_den = 1e-12;
      else
        gamma_den = -1e-12;
    }

    double gamma = (1.0 / alpha) * (gamma_num / gamma_den);
    double beta = (1.0 / alpha) * (X3 - 3.0 * alpha * gamma);

    double disc_val = beta * beta + 4.0 * alpha;
    if (disc_val < 0)
      disc_val = 0;
    double disc = std::sqrt(disc_val);
    double mu1 = (beta - disc) / 2.0;
    double mu2 = (beta + disc) / 2.0;

    double p1 = mu2 / (mu2 - mu1);
    if (std::abs(mu2 - mu1) < 1e-12)
      p1 = 0.5;
    if (p1 < 0)
      p1 = 0;
    else if (p1 > 1)
      p1 = 1;
    double p2 = 1.0 - p1;

    double sigma1_sq = sigma2 - (p1 * mu1 * mu1 + p2 * mu2 * mu2 - mu1 * gamma);
    if (sigma1_sq < 1e-10)
      sigma1_sq = 1e-10;
    double sigma2_sq = sigma1_sq + (mu2 - mu1) * gamma;
    if (sigma2_sq < 1e-10)
      sigma2_sq = 1e-10;

    best_candidate = List::create(
        Named("comp1") = List::create(Named("p") = p1, Named("mu") = mu1 + mu,
                                      Named("sigma") = std::sqrt(sigma1_sq)),
        Named("comp2") = List::create(Named("p") = p2, Named("mu") = mu2 + mu,
                                      Named("sigma") = std::sqrt(sigma2_sq)));
  }

  return best_candidate;
}

// [[Rcpp::export]]
List Recover1DMixture_cpp(NumericVector x, double delta = 0.05) {
  int n = x.length();

  double mu_overall = mean(x);
  double var_overall = var(x) * (n - 1.0) / n; // population variance
  double sigma_overall = std::sqrt(var_overall);

  if (sigma_overall < 1e-12) {
    return List::create(
        Named("comp1") = List::create(
            Named("p") = 0.5, Named("mu") = mu_overall, Named("sigma") = 0.0),
        Named("comp2") = List::create(
            Named("p") = 0.5, Named("mu") = mu_overall, Named("sigma") = 0.0),
        Named("fallback") = true);
  }

  NumericVector x_std = (x - mu_overall) / sigma_overall;

  List moments = estimate_excess_moments_cpp(x_std);
  double mu_std = moments["mu"];         // 0
  double sigma2_std = moments["sigma2"]; // 1
  double X3_std = moments["X3"];
  double X4_std = moments["X4"];
  double X5_std = moments["X5"];
  double X6_std = moments["X6"];

  double f = std::pow(std::log(1.0 / delta) / n, 1.0 / 12.0);

  double eps_noise = 1e-4;
  if (std::abs(X4_std) < eps_noise && X3_std * X3_std < eps_noise) {
    return List::create(Named("comp1") = List::create(
                            Named("p") = 0.5, Named("mu") = mu_overall,
                            Named("sigma") = sigma_overall),
                        Named("comp2") = List::create(
                            Named("p") = 0.5, Named("mu") = mu_overall,
                            Named("sigma") = sigma_overall),
                        Named("fallback") = true);
  }

  double delta_mu_std = 0;
  if (X4_std > 0) {
    delta_mu_std = std::min(std::pow(std::abs(X3_std), 1.0 / 3.0) +
                                std::pow(std::abs(X4_std), 1.0 / 4.0),
                            std::abs(X3_std) / std::sqrt(X4_std));
  } else {
    delta_mu_std = std::pow(std::abs(X3_std), 1.0 / 3.0) +
                   std::pow(std::abs(X4_std), 1.0 / 4.0);
  }

  double delta_sigma2_std = std::sqrt(std::abs(X4_std));

  List best_candidate;

  if (f * f <= delta_mu_std * delta_mu_std) {
    double eps = std::sqrt(std::pow(1.0 / std::max(1e-12, delta_mu_std), 12) *
                           std::log(1.0 / delta) / n);
    best_candidate = RecoverFromMoments_cpp(mu_std, sigma2_std, X3_std, X4_std,
                                            X5_std, X6_std, eps);
  } else if (f * f <= delta_sigma2_std) {
    best_candidate =
        SameMeanRecoverFromMoments_cpp(mu_std, sigma2_std, X4_std, X6_std);
  } else {
    return List::create(Named("comp1") = List::create(
                            Named("p") = 0.5, Named("mu") = mu_overall,
                            Named("sigma") = sigma_overall),
                        Named("comp2") = List::create(
                            Named("p") = 0.5, Named("mu") = mu_overall,
                            Named("sigma") = sigma_overall),
                        Named("fallback") = true);
  }

  // Unstandardize
  List comp1 = best_candidate["comp1"];
  List comp2 = best_candidate["comp2"];

  double m1 = as<double>(comp1["mu"]);
  double s1 = as<double>(comp1["sigma"]);
  double m2 = as<double>(comp2["mu"]);
  double s2 = as<double>(comp2["sigma"]);

  comp1["mu"] = m1 * sigma_overall + mu_overall;
  comp1["sigma"] = s1 * sigma_overall;
  comp2["mu"] = m2 * sigma_overall + mu_overall;
  comp2["sigma"] = s2 * sigma_overall;

  return List::create(Named("comp1") = comp1, Named("comp2") = comp2);
}
