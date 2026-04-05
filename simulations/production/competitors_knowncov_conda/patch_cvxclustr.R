# ------------------------------------------------------------------
# patch_cvxclustr.R
# Downloads and patches cvxclustr to link with BLAS on Windows.
# ------------------------------------------------------------------
download.file('https://cran.r-project.org/src/contrib/Archive/cvxclustr/cvxclustr_1.1.0.tar.gz', 'cvxclustr_1.1.0.tar.gz')
untar('cvxclustr_1.1.0.tar.gz')
# Create Makevars.win with correct BLAS linking tags
writeLines('PKG_LIBS = $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)', 'cvxclustr/src/Makevars.win')
cat("Patched cvxclustr/src/Makevars.win successfully.\n")
