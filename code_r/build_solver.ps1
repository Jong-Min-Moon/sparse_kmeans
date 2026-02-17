
# Build Script for ADMM Solver Rcpp Extension
$ErrorActionPreference = "Stop"

# 1. Setup Rtools PATH
$oldPath = $env:PATH
if ($oldPath -notmatch "rtools") {
    $rtoolsPaths = @(
        "C:\rtools45\ucrt64\bin",
        "C:\rtools45\usr\bin",
        "C:\rtools44\ucrt64\bin",
        "C:\rtools44\usr\bin"
    )
    $validPaths = $rtoolsPaths | Where-Object { Test-Path $_ }
    if ($validPaths) {
        $env:PATH = ($validPaths -join ";") + ";" + $oldPath
        Write-Host "Added Rtools to PATH."
    }
    else {
        Write-Warning "Rtools not found in standard locations. Compilation might fail."
    }
}

# 2. Get Rcpp Flags
Write-Host "Getting Rcpp Flags..."
$Rscript = "C:\Program Files\R\R-4.5.0\bin\x64\Rscript.exe"
$RcppFlags = & $Rscript -e "cat(Rcpp:::CxxFlags())"
Write-Host "Rcpp Flags: $RcppFlags"

# 3. Set Environment Variable for R CMD SHLIB
$env:PKG_CXXFLAGS = $RcppFlags

# 4. Compile
Write-Host "Compiling proj_simplex.cpp..."
$Rcmd = "C:\Program Files\R\R-4.5.0\bin\x64\R.exe"
& $Rcmd CMD SHLIB -o code_r/proj_simplex.dll code_r/proj_simplex.cpp

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilation Successful: code_r/proj_simplex.dll"
}
else {
    Write-Error "Compilation Failed."
}
