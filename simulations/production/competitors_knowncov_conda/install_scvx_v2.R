# Refined scvxclustr installation script for Windows R 4.0.5 (Conda)
pkg_dir <- "scvxclustr-master"

# 1. Environment Configuration
# Prepend Mingw-w64 and MSYS2-compatible paths to ensure g++ and sh are found
conda_prefix <- "C:/Users/jongmin/miniconda3/envs/r_legacy_sim"
mingw_bin <- file.path(conda_prefix, "Library/mingw-w64/bin")
usr_bin <- file.path(conda_prefix, "Library/usr/bin")
lib_bin <- file.path(conda_prefix, "Library/bin")

old_path <- Sys.getenv("PATH")
new_path <- paste(mingw_bin, usr_bin, lib_bin, old_path, sep = ";")
Sys.setenv(PATH = new_path)

# BINPREF must be the empty string for R 4.0.x to correctly use Rtools40 or equivalent MinGW paths
Sys.setenv(BINPREF = "")

# 2. Dependency Check (Snapshot 2021)
snapshot_repo <- "https://cran.microsoft.com/snapshot/2021-04-20/"
options(repos = c(CRAN = snapshot_repo))

if (!requireNamespace("Rcpp", quietly = TRUE)) {
    install.packages("Rcpp", type = "binary")
}
if (!requireNamespace("RcppEigen", quietly = TRUE)) {
    install.packages("RcppEigen", type = "binary")
}

# 3. Patch Makevars.win
# We must ensure $(CXX11) and $(CXX11STD) are used, and point to Rcpp/RcppEigen headers
makevars_path <- file.path(pkg_dir, "src", "Makevars.win")
# Using the format that R's Makeconf understands best
makevars_content <- "
CXX11STD = -std=gnu++11
PKG_CPPFLAGS = -I$(R_HOME)/library/Rcpp/include -I$(R_HOME)/library/RcppEigen/include
PKG_LIBS = $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)
"
writeLines(makevars_content, makevars_path)

# 4. Installation
cat("Installing scvxclustr from:", pkg_dir, "...\n")
install.packages(pkg_dir, repos = NULL, type = "source", verbose = TRUE)

# 5. Verification
if (requireNamespace("scvxclustr", quietly = TRUE)) {
    cat("\nSUCCESS: scvxclustr installed and loaded successfully!\n")
} else {
    stop("\nFAILURE: scvxclustr installation failed.\n")
}
