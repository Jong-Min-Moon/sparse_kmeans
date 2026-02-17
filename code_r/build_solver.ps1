
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

# 2.1 Get OpenMP Flags
Write-Host "Getting OpenMP Flags..."
$Rcmd = "C:\Program Files\R\R-4.5.0\bin\x64\R.exe"
# Start with Rcpp flags
$CxxFlagsVector = $RcppFlags
# On Windows with Rtools, -fopenmp is standard for g++
# We can try to query R CMD config, but we saw it failed for SHLIB_OPENMP_CXXFLAGS.
# However, for R 4.5.0 + Rtools45, -fopenmp is the flag.
$OmpFlags = "-fopenmp"
Write-Host "OpenMP Flags: $OmpFlags"

# 3. Set Environment Variable for R CMD SHLIB
$env:PKG_CXXFLAGS = "$RcppFlags $OmpFlags"
$env:PKG_LIBS = "$OmpFlags"

# 4. Compile
Write-Host "Compiling proj_simplex.cpp..."
# Assuming script is run from code_r or root, but let's be safe. 
# If run from code_r:
if (Test-Path "proj_simplex.cpp") {
    & $Rcmd CMD SHLIB -o proj_simplex.dll proj_simplex.cpp
}
# If run from root:
elseif (Test-Path "code_r/proj_simplex.cpp") {
    & $Rcmd CMD SHLIB -o code_r/proj_simplex.dll code_r/proj_simplex.cpp
}
else {
    Write-Error "Could not find proj_simplex.cpp"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilation Successful."
}
else {
    Write-Error "Compilation Failed."
}
