
import numpy as np
import cvxpy as cp
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.cluster import KMeans

def kmeans_sdp_pengwei(A, k, solver=None, verbose=False):
    """
    Solves the Peng and Wei k-means SDP formulation using cvxpy.
    
    The original MATLAB implementation:
        $\hat{Z} = \arg\max_{Z \in \mathbb{R}^{n \times n}} \langle A, Z \rangle$
        subject to $Z \succeq 0, \mathrm{tr}(Z) = K, Z \mathbf{1}_n = \mathbf{1}_n, Z \geq 0$.

    Args:
        A (np.ndarray): n x n affinity matrix (e.g., Gram matrix X'X or -Distance).
        k (int): Number of clusters.
        solver (str, optional): Solver to use (e.g., cp.SCS, cp.CLARABEL, cp.MOSEK). Defaults to None (let cvxpy choose).
        verbose (bool): standard solver verbosity.

    Returns:
        Z_opt (np.ndarray): The optimal matrix Z.
    """
    n = A.shape[0]
    
    # Define Variable
    Z = cp.Variable((n, n), PSD=True) # Z >= 0 (PSD constraint)

    # Objective: Maximize <A, Z> which is trace(A @ Z)
    # Note: MATLAB code defines D = -A and passes D to SDPNAL+, which minimizes <C, X>.
    # So minimizing <-A, Z> is same as maximizing <A, Z>.
    objective = cp.Maximize(cp.trace(A @ Z))

    # Constraints
    # Flattening Z for the non-negativity check skips the Python bottleneck (as per user's finding)
    constraints = [
        cp.trace(Z) == k,
        cp.sum(Z, axis=1) == 1,
        cp.vec(Z, order='C') >= 0
    ]

    # Problem
    prob = cp.Problem(objective, constraints)

    # Solver
    # Recommended: 'CLARABEL' > 'SCS' (Free solvers)
    if solver is None:
        # Heuristic for best available free solver
        installed_solvers = cp.installed_solvers()
        if 'CLARABEL' in installed_solvers:
             solver = cp.CLARABEL
        else:
             solver = cp.SCS

    try:
        # Default options for SCS if not specified
        if solver == cp.SCS:
            # Matches user's optimized settings
            solver_opts = {'max_iters': 2500, 'eps_abs': 1e-4, 'eps_rel': 1e-4}
        else:
            solver_opts = {}
            
        prob.solve(solver=solver, verbose=verbose, **solver_opts)
    except Exception as e:
        print(f"Solver failed: {e}")
        return None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Warning: Problem status is {prob.status}")
    
    print(f"Solver Status: {prob.status}, Value: {prob.value}")
    if Z.value is not None:
        print(f"Z stats: min={np.min(Z.value)}, max={np.max(Z.value)}, trace={np.trace(Z.value)}")

    return Z.value

def sdp_sol_to_cluster(Z_opt, K):
    """
    Converts the SDP solution Z to cluster labels using spectral decomposition and k-means.
    """
    if Z_opt is None:
        return None
        
    # Extract left singular vectors (or eigenvectors since Z is symmetric PSD)
    # Using eigh for symmetric matrices is generally more stable/faster than svd
    evals, evecs = eigh(Z_opt) 
    
    # Sort eigenvalues/vectors in descending order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    
    # Top K eigenvectors
    U_top_K = evecs[:, :K]
    
    # K-Means on the rows of U_top_K
    kmeans = KMeans(n_clusters=K, n_init=10, max_iter=500, random_state=42).fit(U_top_K)
    
    return kmeans.labels_

