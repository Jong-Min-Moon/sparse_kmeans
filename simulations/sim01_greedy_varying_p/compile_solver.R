# Compilation Script for Linux HPC
# Compiles code_r/proj_simplex.cpp into code_r/proj_simplex.so

# Set Rcpp Flags (capture.output is needed as these print to stdout)
cat("Capturing Rcpp flags...\n")
cxx_flags <- tryCatch(paste(capture.output(Rcpp:::CxxFlags()), collapse = " "), error = function(e) "")
ld_flags <- tryCatch(paste(capture.output(Rcpp:::LdFlags()), collapse = " "), error = function(e) "")

# Set environment variables for R CMD SHLIB
# Adding -fopenmp for parallel performance
Sys.setenv(PKG_CXXFLAGS = paste("-fopenmp", cxx_flags))
Sys.setenv(PKG_LIBS = paste("-fopenmp", ld_flags))

# Helper to compile a C++ file
compile_file <- function(src_name) {
    src_file <- file.path("../../code_r", src_name)
    lib_name <- sub("\\.cpp$", ".so", src_name)
    lib_file <- file.path("../../code_r", lib_name)

    if (!file.exists(src_file)) {
        stop(sprintf("Source file '%s' not found!", src_file))
    }

    # Clean up
    obj_file <- sub("\\.cpp$", ".o", src_file)
    if (file.exists(obj_file)) unlink(obj_file)
    if (file.exists(lib_file)) unlink(lib_file)

    cat(sprintf("Compiling '%s' -> '%s'...\n", src_file, lib_file))
    cmd <- sprintf("R CMD SHLIB -o %s %s", lib_file, src_file)
    status <- system(cmd)

    if (status != 0) stop(sprintf("Compilation of %s failed", src_name))
    cat(sprintf("Compilation of %s successful!\n", src_name))
}

# Compile both
compile_file("proj_simplex.cpp")
compile_file("selection_utils.cpp")
