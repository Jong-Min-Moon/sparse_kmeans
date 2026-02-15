# Helper function for ISEE: Lasso regression to get intercept and residuals

library(glmnet)

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
  if (nrow(predictor) != n) {
    stop("Dimensions of response and predictor do not match.")
  }
  
  # Handle case with constant predictor (glmnet fails if scaling 0 variance)
  # But usually predictor is high dimensional here.
  
  # Fit Lasso with Cross-Validation
  # alpha = 1 for Lasso
  # standardize = TRUE (default, but good to be explicit)
  # intercept = TRUE (default)
  
  # Capture output to silence glmnet's potential print
  # (cv.glmnet is usually silent but good practice)
  cv_fit <- tryCatch({
    cv.glmnet(x = predictor, y = response, alpha = 1, standardize = TRUE, intercept = TRUE)
  }, error = function(e) {
    # Fallback if glmnet fails (e.g., too few samples, constant columns)
    warning(paste("glmnet failed:", e$message, "- falling back to simple mean."))
    return(NULL)
  })
  
  if (is.null(cv_fit)) {
    # Fallback: Intercept = Mean(y), Residual = y - Mean(y) (Null model)
    intercept <- mean(response)
    residual <- response - intercept
    return(list(intercept = intercept, residual = residual))
  }
  
  # Get coefficients at lambda.min (minimum CV error)
  # Could also use lambda.1se for more regularization
  lambda_best <- cv_fit$lambda.min
  
  # Extract coefficients (returns sparse matrix)
  coefs <- coef(cv_fit, s = lambda_best)
  
  # coefs[1] is intercept, valid for glmnet
  intercept <- as.numeric(coefs[1])
  
  # Compute predicted values
  y_pred <- predict(cv_fit, newx = predictor, s = lambda_best)
  
  # Compute residuals
  residual <- as.numeric(response - y_pred)
  
  return(list(intercept = intercept, residual = residual))
}
