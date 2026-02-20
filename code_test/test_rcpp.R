# code_test/test_rcpp.R
# Add Rtools to PATH
old_path <- Sys.getenv("PATH")
rtools_bin <- "C:\\rtools45\\usr\\bin"
ucrt_bin <- "C:\\rtools45\\ucrt64\\bin"
Sys.setenv(PATH = paste(rtools_bin, ucrt_bin, old_path, sep = ";"))

cat("Checking build tools...\n")
cat("Make:", Sys.which("make"), "\n")
cat("G++:", Sys.which("g++"), "\n")

tryCatch(
    {
        library(Rcpp)
        cat("Compiling simple function...\n")
        cppFunction("
    int add_cpp(int x, int y) {
      return x + y;
    }
  ")
        cat("Result of 10 + 20:", add_cpp(10, 20), "\n")
        cat("Rcpp Compilation Successful!\n")
    },
    error = function(e) {
        cat("Rcpp Failed:\n")
        print(e)
        quit(status = 1)
    }
)
