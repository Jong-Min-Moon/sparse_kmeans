# Manual Link and Install for scvxclustr
pkg_dir <- "scvxclustr-master"
src_dir <- file.path(pkg_dir, "src")
conda_prefix <- "C:/Users/jongmin/miniconda3/envs/r_legacy_sim"
r_bin_x64 <- file.path(conda_prefix, "lib/R/bin/x64")
mingw_bin <- file.path(conda_prefix, "Library/mingw-w64/bin")

# 1. Prepare Environment
old_path <- Sys.getenv("PATH")
Sys.setenv(PATH = paste(mingw_bin, r_bin_x64, old_path, sep = ";"))

# 2. Manual Link Command
# We skip the .def file if possible, or use --export-all-symbols
setwd(src_dir)
link_cmd <- paste(
    "g++ -std=gnu++11 -shared -s -o scvxclustr.dll",
    "RcppExports.o admm.o ama.o",
    paste0("-L", r_bin_x64),
    "-lRlapack -lRblas -lgfortran -lm -lquadmath -lR",
    "-Wl,--export-all-symbols"
)

cat("Running manual link command:\n", link_cmd, "\n")
res <- system(link_cmd)

if (res != 0) {
    stop("Manual linking failed!")
}

if (!file.exists("scvxclustr.dll")) {
    stop("DLL was not created!")
}
cat("SUCCESS: scvxclustr.dll created.\n")

# 3. Manual Installation
# We'll use R CMD INSTALL on the parent folder, but we need to make sure 
# it doesn't try to recompile. The --libs-only might work, or just let it 
# see the existing .dll.
setwd("../..")
cat("Installing package with pre-compiled libs...\n")
# We use --no-multiarch and --no-test-load just to get it in there first
install_res <- system(paste0("conda run -n r_legacy_sim R CMD INSTALL --no-multiarch ", pkg_dir))

if (install_res != 0) {
    stop("Final R CMD INSTALL failed!")
}

# 4. Verification
if (requireNamespace("scvxclustr", quietly = TRUE)) {
    cat("\nVERIFIED: scvxclustr is now installed and loadable!\n")
} else {
    stop("\nVerification failed. Package exists but cannot be loaded.\n")
}
