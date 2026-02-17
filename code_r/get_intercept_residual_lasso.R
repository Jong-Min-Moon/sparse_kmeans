# Helper function for ISEE: Lasso regression to get intercept and residuals


# Note: This function requires the 'glmnet' package.


#' Get Intercept and Residuals from Lasso Regression
#'
#' Fits a Lasso regression (using cv.glmnet for lambda selection) and returns
#' the intercept and residuals.
#'
#' @param response Response vector (n x 1)
#' @param predictor Predictor matrix (n x p)
#' @return List containing:
#'   \item{intercept}{Estimated intercept (scalar)}
#'   \item{residual}{Residuals (n x 1 vector)}
#' @export
get_intercept_residual_lasso <- function(response, predictor) {
  # Ensure inputs are valid
  n <- length(response)

  # Data Integrity: Handle data.frame input by converting to matrix
  if (is.data.frame(predictor)) {
    predictor <- as.matrix(predictor)
  }

  if (nrow(predictor) != n) {
    stop("Dimensions of response and predictor do not match.")
  }

  # Handle case with constant predictor (glmnet fails if scaling 0 variance)
  # But usually predictor is high dimensional here.

  # Fit Lasso with Cross-Validation
  # alpha = 1 for Lasso
  # standardize = TRUE (default, but good to be explicit)
  # intercept = TRUE (default)
  # parallel = FALSE (explicitly disable parallelization for worker safety)

  # Capture output to silence glmnet's potential print
  # (cv.glmnet is usually silent but good practice)
  cv_fit <- tryCatch(
    {
      glmnet::cv.glmnet(x = predictor, y = response, alpha = 1, standardize = TRUE, intercept = TRUE, parallel = FALSE)
    },
    error = function(e) {
      # Fallback if glmnet fails (e.g., too few samples, constant columns)
      warning(paste("glmnet failed:", e$message, "- falling back to simple mean."))
      return(NULL)
    }
  )

  if (is.null(cv_fit)) {
    # Fallback: Intercept = Mean(y), Residual = y - Mean(y) (Null model)
    intercept <- mean(response)
    residual <- response - intercept
    return(list(intercept = intercept, residual = residual))
  }

  # Get coefficients at lambda.1se (simplest model within 1 SE of min error)
  # to prevent overfitting.
  lambda_best <- cv_fit$lambda.1se

  # Extract coefficients (returns sparse matrix)
  coefs <- glmnet::coef.glmnet(cv_fit, s = lambda_best)

  # coefs[1] is intercept, valid for glmnet
  intercept <- as.numeric(coefs[1])

  # Optimization: Avoid Double Calculation (predict)
  # Calculate residuals manually: r = y - (Intercept + X * beta)
  # coefs is (p+1) x 1 sparse matrix. Row 1 is Intercept. Rows 2:(p+1) are betas.
  # We need to be careful with sparse matrix multiplication dimensions.

  # Convert sparse coefficients to numeric vector for beta (excluding intercept)
  # Note: coefs is a dgCMatrix.
  beta <- coefs[-1, , drop = FALSE] # Keep as sparse column vector

  # Compute predicted values using sparse multiplication
  # X * beta. Note: predictor might be a dense matrix or sparse matrix.
  # If predictor is dense, result is dense.
  linear_pred <- as.numeric(predictor %*% beta)

  y_pred <- intercept + linear_pred

  # Compute residuals
  residual <- as.numeric(response - y_pred)

  return(list(intercept = intercept, residual = residual))
}
