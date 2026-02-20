
import numpy as np
import time
import concurrent.futures
import os
import cvxpy as cp
from scipy.spatial.distance import pdist, squareform
from kmeans_sdp import kmeans_sdp_pengwei, sdp_sol_to_cluster

def run_single_sim(run_id, p, N, n1, n2, k, center_dist, nonzero_count):
    """
    Runs a single simulation and returns stats.
    """
    try:
        # Seed for reproducibility per run
        np.random.seed(42 + run_id)
        
        mu1 = np.zeros(p)
        mu2 = np.zeros(p)
        mu2[:nonzero_count] = np.sqrt((center_dist**2) / nonzero_count)
        
        X1 = np.random.randn(n1, p) + mu1
        X2 = np.random.randn(n2, p) + mu2
        data_matrix = np.vstack((X1, X2))
        
        D = squareform(pdist(data_matrix, metric='sqeuclidean'))
        A = -D 
        
        start_time = time.time()
        
        # Run Solver - Forcing SCS
        Z_opt = kmeans_sdp_pengwei(A, k, solver=cp.SCS, verbose=False)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if Z_opt is None:
            return {
                "run_id": run_id,
                "duration": duration,
                "accuracy": np.nan,
                "status": "failed"
            }

        # Clustering
        labels = sdp_sol_to_cluster(Z_opt, k)
        
        # Accuracy Check
        true_labels = np.array([0] * n1 + [1] * n2)
        match1 = np.mean(labels == true_labels)
        match2 = np.mean(labels == (1 - true_labels))
        acc = max(match1, match2)
        
        return {
            "run_id": run_id,
            "duration": duration,
            "accuracy": acc,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "run_id": run_id,
            "duration": 0,
            "accuracy": np.nan,
            "status": f"error: {str(e)}"
        }

def run_benchmark():
    # Parameters
    p = 1000 
    N = 200
    n1, n2 = 100, 100
    k = 2
    center_dist = 4
    nonzero_count = 10
    num_runs = 100
    
    # Use max available cores minus one, or at least 1
    max_workers = max(1, os.cpu_count() - 1)
    
    print(f"Starting Parallel Benchmark: P={p}, N={N}, Runs={num_runs}, Cores={max_workers}...")
    
    total_start_time = time.time()
    
    times = []
    accuracies = []
    failed_runs = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_sim, i, p, N, n1, n2, k, center_dist, nonzero_count) 
            for i in range(num_runs)
        ]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            
            if res["status"] == "success":
                times.append(res["duration"])
                accuracies.append(res["accuracy"])
                # print(f"  [Run {res['run_id']}] Time: {res['duration']:.2f}s | Acc: {res['accuracy']*100:.1f}%")
            else:
                failed_runs += 1
                print(f"  [Run {res['run_id']}] Failed: {res['status']}")
            
            # Simple progress header every 10 runs
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs...")

    total_end_time = time.time()
    
    print("\n--- Benchmark Summary ---")
    print(f"Total Wall-Clock Time: {(total_end_time - total_start_time):.2f} seconds")
    
    if times:
        avg_time = np.mean(times)
        avg_acc = np.mean(accuracies)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"Successful Runs: {len(times)}/{num_runs}")
        print(f"Average Runtime: {avg_time:.4f} seconds")
        print(f"Min Runtime:     {min_time:.4f} seconds")
        print(f"Max Runtime:     {max_time:.4f} seconds")
        print(f"Average Accuracy: {avg_acc*100:.2f}%")
        
        if avg_time < 5.0:
            print("SUCCESS: Average runtime is less than 5 seconds.")
        else:
            print("WARNING: Average runtime executed 5 seconds.")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    run_benchmark()
