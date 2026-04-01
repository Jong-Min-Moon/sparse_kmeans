# ------------------------------------------------------------------
# install_legacy.R
# Final attempt at installing legacy R packages in r_legacy_sim.
# ------------------------------------------------------------------

# 1. Force R to use Conda's build tools on Windows
env_path <- Sys.getenv("CONDA_PREFIX")
mingw_bin <- normalizePath(file.path(env_path, "Library", "mingw-w64", "bin"), winslash = "/")
usr_bin   <- normalizePath(file.path(env_path, "Library", "usr", "bin"), winslash = "/")

cat(sprintf("Configuring toolchain from:\n  %s\n  %s\n", mingw_bin, usr_bin))

# Update PATH to prioritize Conda's tools (both mingw and m2/usr)
new_path <- paste(mingw_bin, usr_bin, Sys.getenv("PATH"), sep = ";")
Sys.setenv(PATH = new_path)
Sys.setenv(MAKE = "make") # We already aliased mingw32-make to make

# 2. Use a stable MRAN snapshot from 2021 (R 4.0.5 era)
repos <- c(CRAN = "https://packagemanager.posit.co/cran/2021-04-20")
options(repos = repos)

# 3. Install Standard Dependencies (Force Binaries to avoid toolchain issues)
cat("Installing standard binaries...\n")
std_pkgs <- c("dplyr", "tidyr", "foreach", "doParallel", "sparcl", "clue", "MASS", "remotes", "methods", "igraph", "mclust", "gglasso", "Rcpp", "RcppEigen")
install.packages(std_pkgs, repos = repos, type = "binary")

# 4. Install bclust from Archive (v1.5)
if (!requireNamespace("bclust", quietly = TRUE)) {
  cat("Installing bclust 1.5...\n")
  url_bclust <- "https://cran.r-project.org/src/contrib/Archive/bclust/bclust_1.5.tar.gz"
  install.packages(url_bclust, repos = NULL, type = "source")
}

# 5. Install patched cvxclustr (already unzipped and patched via patch_cvxclustr.R)
if (!requireNamespace("cvxclustr", quietly = TRUE)) {
  cat("Installing patched cvxclustr...\n")
  # Use the folder directly if it exists
  if (dir.exists("cvxclustr")) {
    system("make -v") # Debug check
    install.packages("cvxclustr", repos = NULL, type = "source")
  }
}

# 6. Install scvxclustr from GitHub
if (!requireNamespace("scvxclustr", quietly = TRUE)) {
  cat("Installing scvxclustr from GitHub...\n")
  remotes::install_github("elong0527/scvxclustr", dependencies = FALSE, upgrade = "never")
}

# 7. Verification
cat("\nFinal Check:\n")
all_pkgs <- c(std_pkgs, "bclust", "cvxclustr", "scvxclustr")
results <- data.frame(Package = all_pkgs, Status = "FAILED", stringsAsFactors = FALSE)
for (i in seq_along(all_pkgs)) {
  if (requireNamespace(all_pkgs[i], quietly = TRUE)) {
    results$Status[i] <- "OK"
  }
}
print(results)
