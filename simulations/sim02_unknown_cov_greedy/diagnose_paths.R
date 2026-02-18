cat("--- Path Diagnostic ---\n")
cat("CWD:", getwd(), "\n")
lib_name <- "selection_utils"
ext <- if (.Platform$OS.type == "windows") ".dll" else ".so"
possibilities <- c(
    paste0("code_r/", lib_name, ext),
    paste0("../../code_r/", lib_name, ext),
    paste0("../code_r/", lib_name, ext)
)
for (p in possibilities) {
    cat(sprintf("Checking '%s': %s\n", p, file.exists(p)))
}
cat("--- End Diagnostic ---\n")
