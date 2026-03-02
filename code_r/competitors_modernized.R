library(sparcl)

#' Run Witten's Sparse K-Means
#' @param X n x p data matrix
#' @param K number of clusters (default 2)
#' @param seed optional random seed
run_witten <- function(X, K = 2, seed = NULL) {
    if (!is.null(seed)) {
        set.seed(seed)
    }

    # Ensure X is n x p (if it's p x n like in the data generator output with p=400, n=200)
    if (nrow(X) > ncol(X) && ncol(X) > 0) {
        # It might be p x n, let's just transpose it.
        X <- t(X)
    }

    # Scale data as done in original code
    x_scaled <- scale(X, TRUE, TRUE)

    # The original code uses wbounds = seq(3, 7, length.out=15)
    suppressWarnings({
        suppressMessages({
            km.perm <- KMeansSparseCluster.permute(x_scaled, K = K, wbounds = seq(3, 7, length.out = 15), nperms = 5)
            km.out <- KMeansSparseCluster(x_scaled, K = K, wbounds = km.perm$bestw)
        })
    })

    return(km.out[[1]]$Cs)
}

# --- Arias-Castro Sparse K-Means (Hill Climbing) ---
# Helper for Hill Climbing
Alternate <- function(X, k, tot, initial_set, s, itermax, threshold) {
    p <- dim(X)[2]
    set0 <- initial_set
    set1 <- c(0)
    iternum <- 0

    while (iternum <= itermax && length(setdiff(set1, set0)) + length(setdiff(set0, set1)) > threshold) {
        clustering <- kmeans(X[, set0, drop = FALSE],
            iter.max = 20, centers = k,
            algorithm = "Hartigan-Wong", trace = 0, nstart = 2
        )
        result <- clustering$cluster
        group <- list()
        cond <- TRUE

        for (j in seq_len(k)) {
            group[[j]] <- which(result == j)
            cond <- cond && length(group[[j]]) > 1
        }

        center <- NULL
        wcss <- rep(0, p)

        if (cond) {
            for (j in seq_len(k)) {
                center <- rbind(center, colMeans(X[group[[j]], , drop = FALSE]))
                Xc <- t(apply(X[group[[j]], , drop = FALSE], 1, function(x) x - center[j, ]))
                wcss <- wcss + apply(Xc, 2, function(x) {
                    sum(x^2)
                })
            }
            iternum <- iternum + 1
        }

        set1 <- set0
        set0 <- which(rank((tot - wcss) / tot, ties.method = "random") > p - s)
    }

    out <- list(final_set = set0, iternum = iternum, result = result, betweenss = clustering$betweenss)
    return(out)
}

# Helper for Hill Climbing
hill_climb <- function(X, k, nbins = 50, nperms = 25, itermax = 100, threshold = 1e-5) {
    n <- dim(X)[1]
    p <- dim(X)[2]

    center0 <- colMeans(X)
    Xc0 <- t(apply(X, 1, function(x) x - center0))
    tot <- apply(Xc0, 2, function(x) {
        sum(x^2)
    })

    permx <- list()
    for (i in 1:nperms) {
        permx[[i]] <- matrix(NA, nrow = n, ncol = p)
        for (j in 1:p) permx[[i]][, j] <- sample(X[, j])
    }

    wcss <- rep(0, p)
    for (j in 1:p) {
        if (length(unique(X[, j])) < k) {
            wcss[j] <- tot[j]
        } else {
            clustering <- kmeans(X[, j], iter.max = 10, centers = k, algorithm = "Hartigan-Wong", trace = 0)
            wcss[j] <- clustering$tot.withinss
        }
    }

    stepsize <- p / nbins
    rank0 <- rank((tot - wcss) / tot, ties.method = "random")

    tots <- NULL
    permtots <- matrix(NA, nrow = nbins, ncol = nperms)
    tots <- c(tots, sum(tot - wcss))
    outs <- list()

    for (i in 2:nbins) {
        s <- floor(p - (i - 1) * stepsize)
        initial_set <- which(rank0 > p - s)
        out <- Alternate(X, k, tot, initial_set, s, itermax, threshold)
        outs[[i]] <- out
        tots <- c(tots, out$betweenss)

        for (t in 1:nperms) {
            permresult <- kmeans(permx[[t]][, out$final_set, drop = FALSE], iter.max = 20, centers = k, algorithm = "Hartigan-Wong", trace = 0)
            permtots[i, t] <- permresult$betweenss
        }
    }

    gaps <- (log(tots) - apply(log(permtots), 1, mean))
    idx <- which.max(gaps)[1]

    all_info <- list(
        idx = idx,
        outs = outs,
        gaps = gaps,
        feature_set = outs[[idx]]$final_set,
        best_result = outs[[idx]]$result
    )

    return(all_info)
}

#' Run Arias-Castro et al Sparse K-Means (Hill Climbing)
#' @param X n x p data matrix
#' @param K number of clusters (default 2)
#' @param seed optional random seed
run_arias <- function(X, K = 2, seed = NULL) {
    if (!is.null(seed)) {
        set.seed(seed)
    }

    # Ensure X is n x p (if it's p x n like in the data generator output with p=400, n=200)
    if (nrow(X) > ncol(X) && ncol(X) > 0) {
        # It might be p x n, let's just transpose it.
        X <- t(X)
    }

    res <- hill_climb(X, K, nbins = 50, nperms = 25, itermax = 100, threshold = 1e-5)
    return(res$best_result)
}
