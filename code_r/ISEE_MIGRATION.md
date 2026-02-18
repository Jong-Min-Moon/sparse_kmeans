# ISEE_bicluster.R - Function Replacement Summary

## Change Made
The default `ISEE_bicluster()` function has been **replaced** with the Post-Lasso implementation based on comprehensive empirical evidence.

## Available Implementations

1. **`ISEE_bicluster()`** (NEW DEFAULT) → Post-Lasso (Two-Stage)
   - Uses Post-Lasso: Lasso selection + OLS refit
   - Best theoretical recovery (shared slopes, unbiased)
   - 63% better residual recovery vs original
   - Robust to heavy-tailed distributions

2. **`ISEE_bicluster_original()`** (DEPRECATED)
   - Separate slopes per cluster (theoretically incorrect)
   - Kept for backward compatibility

3. **`ISEE_bicluster_stacked()`** (ALTERNATIVE)
   - Shared slopes with Lasso estimation
   - Good but biased by Lasso shrinkage

4. **`ISEE_bicluster_postlasso()`** (EXPLICIT)
   - Same as new default `ISEE_bicluster()`
   - Use this if you want to be explicit about method

## Migration Guide

**No code changes needed!** All existing calls to `ISEE_bicluster()` will now automatically use the superior Post-Lasso implementation.

If you need the old behavior for comparison:
```r
# Old behavior
res_old <- ISEE_bicluster_original(X, cluster_est)

# New default (Post-Lasso)
res_new <- ISEE_bicluster(X, cluster_est)
```

## Performance Summary (100 Replications, n=200, p=100)

| Metric | Original | Post-Lasso | Improvement |
|--------|----------|------------|-------------|
| Intercept MSE | 0.0323 | 0.0192 | **40%** |
| Residual MSE | 0.181 | 0.0673 | **63%** |
| Cov Error | 0.354 | 0.223 | **37%** |
