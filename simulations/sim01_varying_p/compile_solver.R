# Helper script to compile proj_simplex.cpp on HPC
# Handles Rcpp flags and OpenMP correctly

cpp_file <- "proj_simplex.cpp"
output_lib <- "proj_simplex.so"

# Check if file exists
if (!file.exists(cpp_file)) {
    stop(paste("File not found:", cpp_file))
}

# Clean old files
file.remove(list.files(pattern = "\\.(o|so|dll)$"))

# Get Rcpp flags (these functions print to stdout, so we capture them)
cxx_flags <- capture.output(Rcpp:::CxxFlags())
ld_flags <- capture.output(Rcpp:::LdFlags())

# Set environment variables for R CMD SHLIB
Sys.setenv(PKG_CXXFLAGS = paste("-fopenmp", paste(cxx_flags, collapse = " ")))
Sys.setenv(PKG_LIBS = paste("-fopenmp", paste(ld_flags, collapse = " ")))

cat("Compiling", cpp_file, "with OpenMP and Rcpp...\n")
cat("PKG_CXXFLAGS:", Sys.getenv("PKG_CXXFLAGS"), "\n")
cat("PKG_LIBS:", Sys.getenv("PKG_LIBS"), "\n")

# Run compilation
res <- system(sprintf("R CMD SHLIB -o %s %s", output_lib, cpp_file))

if (res != 0) {
    stop("Compilation failed")
} else {
    cat("Successfully compiled", output_lib, "\n")
}
