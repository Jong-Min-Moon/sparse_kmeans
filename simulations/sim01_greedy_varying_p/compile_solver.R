# Compilation Script for Linux HPC
# Compiles code_r/proj_simplex.cpp into code_r/proj_simplex.so

# Set source and target paths
src_file <- "../../code_r/proj_simplex.cpp"
lib_file <- "../../code_r/proj_simplex.so"

# Check if source exists
if (!file.exists(src_file)) {
    stop(sprintf("Source file '%s' not found!", src_file))
}

# Clean up potential artifacts (especially Windows-compiled objects transferred via SCP)
obj_file <- sub("\\.cpp$", ".o", src_file)
if (file.exists(obj_file)) {
    cat(sprintf("Removing existing object file '%s'...\n", obj_file))
    unlink(obj_file)
}
if (file.exists(lib_file)) {
    unlink(lib_file)
}

# Set Rcpp Flags
cat("Setting Rcpp flags...\n")
cxx_flags <- tryCatch(paste(capture.output(Rcpp:::CxxFlags()), collapse = " "), error = function(e) "")
ld_flags <- tryCatch(paste(capture.output(Rcpp:::LdFlags()), collapse = " "), error = function(e) "")

if (nchar(cxx_flags) == 0) warning("Rcpp::CxxFlags() returned empty string.")
if (nchar(ld_flags) == 0) warning("Rcpp::LdFlags() returned empty string.")

# Set environment variables for R CMD SHLIB
Sys.setenv(PKG_CXXFLAGS = cxx_flags)
Sys.setenv(PKG_LIBS = ld_flags)

cat(sprintf("Compiling '%s' -> '%s'...\n", src_file, lib_file))

# Compile using R CMD SHLIB
cmd <- sprintf("R CMD SHLIB -o %s %s", lib_file, src_file)
status <- system(cmd)

if (status == 0) {
    cat("Compilation successful!\n")
} else {
    stop("Compilation failed with status ", status)
}
