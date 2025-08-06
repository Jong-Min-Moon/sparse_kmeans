rep_start = 1
rep_end = 200
dimension  = 50
separation = 4
library(sparcl)
library(RSQLite)
source("D:/GitHub/sparse_kmeans/experiment/25_04_03_2025/competitor.R")
source("D:/GitHub/sparse_kmeans/experiment/25_04_03_2025/data_generation.R")
source("D:/GitHub/sparse_kmeans/experiment/25_04_03_2025/get_proportional_subsample")
 
x <- read.csv("D:/GitHub/sparse_kmeans/experiment/28_07_30_2025/real/selected_data_small.csv")
y <- read.csv("D:/GitHub/sparse_kmeans/experiment/28_07_30_2025/real/selected_labels_small.csv")
 



      cluster_est = sparse_kmeans_ell1_ell2(x)
    evaluate_clustering(cluster_est, y)

    cluster_arias = sparse_kmeans_hillclimb(x)
    evaluate_clustering(cluster_arias, y )
    
    acc_witten = 0
    acc_arias = 0
    for (i in 1:30){
      data = get_proportional_subsample(x, y, 45, i)  
      sol_witten = sparse_kmeans_ell1_ell2(data$X)
      acc_witten = acc_witten + evaluate_clustering(sol_witten, data$y)/30
      
      sol_arias = sparse_kmeans_hillclimb(data$X)
      acc_arias = acc_arias + evaluate_clustering(sol_arias, data$y)/30
    }
    
    
    

    