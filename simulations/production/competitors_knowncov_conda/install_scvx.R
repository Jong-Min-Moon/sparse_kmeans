# Targeted install script for scvxclustr in R 4.0.5 (Conda/Windows)
pkg_name <- "scvxclustr-master"

# 1. Identify Toolchain Paths
conda_prefix <- "C:/Users/jongmin/miniconda3/envs/r_legacy_sim"
mingw_bin <- file.path(conda_prefix, "Library/mingw-w64/bin")
usr_bin <- file.path(conda_prefix, "Library/usr/bin")

# 2. Set environment variables to ensure g++ is found by sh.exe
# We prepend mingw_bin and usr_bin to PATH
old_path <- Sys.getenv("PATH")
new_path <- paste(mingw_bin, usr_bin, old_path, sep = ";")
Sys.setenv(PATH = new_path)

# 3. Set BINPREF - critically important for R 4.0.5 x64 on Windows
# In R 4.0, if BINPREF is not set correctly, it fails to find g++
# We point it to the prefix of the executables (empty if they are directly in path)
Sys.setenv(BINPREF = "") 

# 4. Install Rcpp and RcppEigen from 2021 snapshot if not already present
# (scvxclustr depends on these)
snapshot_repo <- "https://cran.microsoft.com/snapshot/2021-04-20/"
if (!requireNamespace("Rcpp", quietly = TRUE)) {
    install.packages("Rcpp", repos = snapshot_repo, type = "binary")
}
if (!requireNamespace("RcppEigen", quietly = TRUE)) {
    install.packages("RcppEigen", repos = snapshot_repo, type = "binary")
}

# 5. Fix/Create src/Makevars.win to ensure C++11 is explicitly requested correctly
# or to bypass standard R deduction if problematic
makevars_path <- file.path(pkg_name, "src", "Makevars.win")
makevars_content <- "CXX_STD = CXX11\nPKG_CPPFLAGS = -I$(R_HOME)/library/Rcpp/include -I$(R_HOME)/library/RcppEigen/include\n"
writeLines(makevars_content, makevars_path)

# 6. Attempt installation
cat("Attempting to install scvxclustr from:", pkg_name, "\n")
install.packages(pkg_name, repos = NULL, type = "source", verbose = TRUE)

# Verification
if (requireNamespace("scvxclustr", quietly = TRUE)) {
    cat("SUCCESS: scvxclustr installed successfully!\n")
} else {
    stop("FAILED to install scvxclustr. Check logs above.\n")
}
